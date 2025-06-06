from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
    Union,
)

import aiohttp
import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

if TYPE_CHECKING:
    from xinference.client import RESTfulChatModelHandle, RESTfulGenerateModelHandle
    from xinference.model.llm.core import LlamaCppGenerateConfig


class Xinference(LLM):
    """`Xinference` large-scale model inference service.

    To use, you should have the xinference library installed:

    .. code-block:: bash

       pip install "xinference[all]"

    If you're simply using the services provided by Xinference, you can utilize the xinference_client package:

    .. code-block:: bash

        pip install xinference_client

    Check out: https://github.com/xorbitsai/inference
    To run, you need to start a Xinference supervisor on one server and Xinference workers on the other servers

    Example:
        To start a local instance of Xinference, run

        .. code-block:: bash

           $ xinference

        You can also deploy Xinference in a distributed cluster. Here are the steps:

        Starting the supervisor:

        .. code-block:: bash

           $ xinference-supervisor

        Starting the worker:

        .. code-block:: bash

           $ xinference-worker

    Then, launch a model using command line interface (CLI).

    Example:

    .. code-block:: bash

       $ xinference launch -n orca -s 3 -q q4_0

    It will return a model UID. Then, you can use Xinference with LangChain.

    Example:

    .. code-block:: python

        from langchain_xinference import Xinference

        llm = Xinference(
            server_url="http://0.0.0.0:9997",
            model_uid = {model_uid} # replace model_uid with the model UID return from launching the model
        )

        llm.invoke(
            prompt="Q: where can we visit in the capital of France? A:",
            generate_config={"max_tokens": 1024, "stream": True},
        )

    Example:

    .. code-block:: python

        from langchain_xinference import Xinference
        from langchain.prompts import PromptTemplate

        llm = Xinference(
            server_url="http://0.0.0.0:9997",
            model_uid={model_uid}, # replace model_uid with the model UID return from launching the model
            stream=True
        )
        prompt = PromptTemplate(
            input=['country'],
            template="Q: where can we visit in the capital of {country}? A:"
        )
        chain = prompt | llm
        chain.stream(input={'country': 'France'})


    To view all the supported builtin models, run:

    .. code-block:: bash

        $ xinference list --all

    """  # noqa: E501

    client: Optional[Any] = None
    server_url: Optional[str]
    """URL of the xinference server"""
    model_uid: Optional[str]
    """UID of the launched model"""
    model_kwargs: Dict[str, Any]
    """Keyword arguments to be passed to xinference.LLM"""

    def __init__(
        self,
        server_url: Optional[str] = None,
        model_uid: Optional[str] = None,
        api_key: Optional[str] = None,
        **model_kwargs: Any,
    ):
        try:
            from xinference.client import RESTfulClient
        except ImportError:
            try:
                from xinference_client import RESTfulClient
            except ImportError as e:
                raise ImportError(
                    "Could not import RESTfulClient from xinference. Please install it"
                    " with `pip install xinference` or `pip install xinference_client`."
                ) from e

        model_kwargs = model_kwargs or {}

        super().__init__(
            **{  # type: ignore[arg-type]
                "server_url": server_url,
                "model_uid": model_uid,
                "model_kwargs": model_kwargs,
            }
        )

        if self.server_url is None:
            raise ValueError("Please provide server URL")

        if self.model_uid is None:
            raise ValueError("Please provide the model UID")

        self._headers: Dict[str, str] = {}
        self._cluster_authed = False
        self._check_cluster_authenticated()
        if api_key is not None and self._cluster_authed:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self.client = RESTfulClient(server_url, api_key)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "xinference"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"server_url": self.server_url},
            **{"model_uid": self.model_uid},
            **{"model_kwargs": self.model_kwargs},
        }

    def _check_cluster_authenticated(self) -> None:
        url = f"{self.server_url}/v1/cluster/auth"
        response = requests.get(url)
        if response.status_code == 404:
            self._cluster_authed = False
        else:
            if response.status_code != 200:
                raise RuntimeError(f"Failed to get cluster information, detail: {response.json()['detail']}")
            response_data = response.json()
            self._cluster_authed = bool(response_data["auth"])

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the xinference model and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: Optional list of stop words to use when generating.
            generate_config: Optional dictionary for the configuration used for
                generation.

        Returns:
            The generated string by the model.
        """
        if self.client is None:
            raise ValueError("Client is not initialized!")
        model = self.client.get_model(self.model_uid)

        generate_config: "LlamaCppGenerateConfig" = kwargs.get("generate_config", {})

        generate_config = {**self.model_kwargs, **generate_config}

        if stop:
            generate_config["stop"] = stop

        if generate_config and generate_config.get("stream"):
            combined_text_output = ""
            for token in self._stream_generate(
                model=model,
                prompt=prompt,
                run_manager=run_manager,
                generate_config=generate_config,
            ):
                combined_text_output += token
            return combined_text_output

        else:
            completion = model.generate(prompt=prompt, generate_config=generate_config)
            return completion["choices"][0]["text"]

    def _stream_generate(
        self,
        model: Union["RESTfulGenerateModelHandle", "RESTfulChatModelHandle"],
        prompt: str,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        generate_config: Optional["LlamaCppGenerateConfig"] = None,
    ) -> Generator[str, None, None]:
        """
        Args:
            prompt: The prompt to use for generation.
            model: The model used for generation.
            stop: Optional list of stop words to use when generating.
            generate_config: Optional dictionary for the configuration used for
                generation.

        Yields:
            A string token.
        """
        streaming_response = model.generate(prompt=prompt, generate_config=generate_config)
        for chunk in streaming_response:
            if isinstance(chunk, dict):
                choices = chunk.get("choices", [])
                if choices:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        token = choice.get("text", "")
                        log_probs = choice.get("logprobs")
                        if run_manager:
                            run_manager.on_llm_new_token(token=token, verbose=self.verbose, log_probs=log_probs)
                        yield token

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        generate_config = kwargs.get("generate_config", {})
        generate_config = {**self.model_kwargs, **generate_config}
        if stop:
            generate_config["stop"] = stop
        for stream_resp in self._create_generate_stream(prompt, generate_config):
            if stream_resp:
                chunk = self._stream_response_to_generation_chunk(stream_resp)
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    def _create_generate_stream(
        self, prompt: str, generate_config: Optional[Dict[str, List[str]]] = None
    ) -> Iterator[str]:
        if self.client is None:
            raise ValueError("Client is not initialized!")
        model = self.client.get_model(self.model_uid)
        yield from model.generate(prompt=prompt, generate_config=generate_config)

    @staticmethod
    def _stream_response_to_generation_chunk(
        stream_response: str,
    ) -> GenerationChunk:
        """Convert a stream response to a generation chunk."""
        token = ""
        if isinstance(stream_response, dict):
            choices = stream_response.get("choices", [])
            if choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    token = choice.get("text", "")

                    return GenerationChunk(
                        text=token,
                        generation_info=dict(
                            finish_reason=choice.get("finish_reason", None),
                            logprobs=choice.get("logprobs", None),
                        ),
                    )
                else:
                    raise TypeError("choice type error!")
            else:
                return GenerationChunk(text=token)
        else:
            raise TypeError("stream_response type error!")

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        generate_config = kwargs.get("generate_config", {})
        generate_config = {**self.model_kwargs, **generate_config}
        if stop:
            generate_config["stop"] = stop
        async for stream_resp in self._acreate_generate_stream(prompt, generate_config):
            if stream_resp:
                chunk = self._stream_response_to_generation_chunk(stream_resp)
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    async def _acreate_generate_stream(
        self, prompt: str, generate_config: Optional[Dict[str, List[str]]] = None
    ) -> AsyncIterator[str]:
        request_body: Dict[str, Any] = {"model": self.model_uid, "prompt": prompt}
        if generate_config is not None:
            for key, value in generate_config.items():
                request_body[key] = value

        stream = bool(generate_config and generate_config.get("stream"))
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=f"{self.server_url}/v1/completions",
                json=request_body,
            ) as response:
                if response.status != 200:
                    if response.status == 404:
                        raise FileNotFoundError("astream call failed with status code 404.")
                    else:
                        optional_detail = response.text
                        raise ValueError(
                            f"astream call failed with status code {response.status}. Details: {optional_detail}"
                        )

                async for line in response.content:
                    if not stream:
                        yield json.loads(line)
                    else:
                        json_str = line.decode("utf-8")
                        if line.startswith(b"data:"):
                            json_str = json_str[len(b"data:") :].strip()
                            if not json_str:
                                continue
                            yield json.loads(json_str)
