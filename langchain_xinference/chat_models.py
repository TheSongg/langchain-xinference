from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

if TYPE_CHECKING:
    from xinference.client.handlers import AsyncChatModelHandle, ChatModelHandle
    from xinference.model.llm.core import LlamaCppGenerateConfig


class ChatXinference(BaseChatModel):
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

    It will return a model UID. Then, you can use ChatXinference with LangChain.

    Example:

    .. code-block:: python

        from langchain_xinference import ChatXinference

        llm = ChatXinference(
            server_url="http://0.0.0.0:9997",
            model_uid={model_uid},  # replace model_uid with the model UID return from launching the model
        )

        llm.invoke(
            input="Q: where can we visit in the capital of France? A:",
            generate_config={"max_tokens": 1024, "stream": True},
        )

    Example:

    .. code-block:: python

        from langchain_xinference import ChatXinference
        from langchain.prompts import PromptTemplate

        llm = ChatXinference(
            server_url="http://0.0.0.0:9997",
            model_uid={model_uid},  # replace model_uid with the model UID return from launching the model
        )
        prompt = PromptTemplate(input=["country"], template="Q: where can we visit in the capital of {country}? A:")
        chain = prompt | llm
        chain.invoke(input={"country": "France"})

        chain.stream(input={"country": "France"})  #  streaming data


    To view all the supported builtin models, run:

    .. code-block:: bash

        $ xinference list --all

    """  # noqa: E501

    client: Optional[Any] = None
    async_client: Optional[Any] = None
    server_url: Optional[str]
    """URL of the xinference server"""
    model_uid: Optional[str]
    """UID of the launched model"""
    model_kwargs: Dict[str, Any]
    """Keyword arguments to be passed to xinference.LLM"""
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[List[str] | str] = None

    def __init__(
        self,
        server_url: Optional[str] = None,
        model_uid: Optional[str] = None,
        api_key: Optional[str] = None,
        **model_kwargs: Any,
    ):
        try:
            from xinference.client import AsyncRESTfulClient, RESTfulClient
        except ImportError:
            try:
                from xinference_client import AsyncRESTfulClient, RESTfulClient
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
        try:
            self.async_client = AsyncRESTfulClient(server_url, api_key)
        except RuntimeError:
            self.async_client = None

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "xinference-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
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

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.client is None:
            raise ValueError("Client is not initialized!")

        model = self.client.get_model(self.model_uid)
        generate_config: "LlamaCppGenerateConfig" = kwargs.get("generate_config", {})
        generate_config = {**self.model_kwargs, **generate_config}

        if stop:
            generate_config["stop"] = stop

        final_chunk = self._chat_with_aggregation(
            model=model,
            messages=messages,
            run_manager=run_manager,
            verbose=self.verbose,
            generate_config=generate_config,
        )

        result = AIMessage(
            content=final_chunk.message.content,
            additional_kwargs=final_chunk.message.additional_kwargs,
            tool_calls=final_chunk.message.tool_calls,
        )

        chat_generation = ChatGeneration(
            message=result,
            generation_info=final_chunk.generation_info,
        )

        return ChatResult(generations=[chat_generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.async_client is None:
            raise ValueError("Client is not initialized!")

        model = await self.async_client.get_model(self.model_uid)
        generate_config: "LlamaCppGenerateConfig" = kwargs.get("generate_config", {})
        generate_config = {**self.model_kwargs, **generate_config}

        if stop:
            generate_config["stop"] = stop

        final_chunk = await self._achat_with_aggregation(
            model=model,
            messages=messages,
            run_manager=run_manager,
            verbose=self.verbose,
            generate_config=generate_config,
        )

        result = AIMessage(
            content=final_chunk.message.content,
            additional_kwargs=final_chunk.message.additional_kwargs,
            tool_calls=final_chunk.message.tool_calls,
        )

        chat_generation = ChatGeneration(
            message=result,
            generation_info=final_chunk.generation_info,
        )

        return ChatResult(generations=[chat_generation])

    def _chat_with_aggregation(
        self,
        model: Union["ChatModelHandle"],
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        generate_config: Optional["LlamaCppGenerateConfig"] = None,
    ) -> ChatGenerationChunk:
        tools = self._choice_tools(tool_choice=self.tool_choice)
        response = model.chat(
            messages=convert_to_openai_messages(messages),
            tools=tools,
            generate_config=generate_config,
        )
        if isinstance(response, dict):
            chunk = self._chat_response_to_chat_generation_chunk(response["choices"][0])
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=verbose,
                )
            return chunk

        final_chunk: Optional[ChatGenerationChunk] = None
        for stream_resp in response:
            if stream_resp:
                chunk = self._chat_response_to_chat_generation_chunk(stream_resp["choices"][0])
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from xinference stream.")

        return final_chunk

    async def _achat_with_aggregation(
        self,
        model: Union["AsyncChatModelHandle"],
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        generate_config: Optional["LlamaCppGenerateConfig"] = None,
    ) -> ChatGenerationChunk:
        tools = self._choice_tools(tool_choice=self.tool_choice)
        response = await model.chat(
            messages=convert_to_openai_messages(messages),
            tools=tools,
            generate_config=generate_config,
        )
        if isinstance(response, dict):
            response = response
            chunk = self._chat_response_to_chat_generation_chunk(response["choices"][0])
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=verbose,
                )
            return chunk

        final_chunk: Optional[ChatGenerationChunk] = None
        async for stream_resp in response:
            if stream_resp:
                chunk = self._chat_response_to_chat_generation_chunk(stream_resp["choices"][0])
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from xinference stream.")

        return final_chunk

    def _chat_response_to_chat_generation_chunk(
        self,
        stream_response: Dict[str, Any],
    ) -> ChatGenerationChunk:
        if stream_response.get("finish_reason") in ["stop", "tool_calls", "length"]:
            generation_info = stream_response
        else:
            generation_info = None

        if "message" in stream_response:
            message = stream_response["message"]
        elif "delta" in stream_response:
            message = stream_response["delta"]
        else:
            raise ValueError("Received unsupported response format from xinference.")
        if message["content"] is None:
            message["content"] = ""
        additional_kwargs = {}
        if "reasoning_content" in message:
            additional_kwargs["reasoning_content"] = message["reasoning_content"]
        if stream_response.get("finish_reason") == "tool_calls":
            tool_calls = message["tool_calls"]
            built_tool_calls = self._build_tool_calls(tool_calls)
            additional_kwargs["tool_calls"] = tool_calls
            chat_chunk = AIMessageChunk(
                content=message["content"],
                additional_kwargs=additional_kwargs,
                tool_call_chunks=built_tool_calls,
            )
        else:
            chat_chunk = AIMessageChunk(
                content=message["content"],
                additional_kwargs=additional_kwargs,
            )
        return ChatGenerationChunk(message=chat_chunk, generation_info=generation_info)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.client is None:
            raise ValueError("Client is not initialized!")

        model = self.client.get_model(self.model_uid)

        generate_config = kwargs.get("generate_config", {})
        if "stream" not in generate_config or not generate_config.get("stream", False):
            generate_config["stream"] = True
        generate_config = {**self.model_kwargs, **generate_config}
        if stop:
            generate_config["stop"] = stop

        tools = self._choice_tools(tool_choice=self.tool_choice)
        response = model.chat(
            messages=convert_to_openai_messages(messages),
            tools=tools,
            generate_config=generate_config,
        )

        for stream_resp in response:
            if stream_resp:
                chunk = self._chat_response_to_chat_generation_chunk(stream_resp["choices"][0])
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.client is None:
            raise ValueError("Client is not initialized!")

        model = await self.async_client.get_model(self.model_uid)

        generate_config = kwargs.get("generate_config", {})
        if "stream" not in generate_config or not generate_config.get("stream", False):
            generate_config["stream"] = True
        generate_config = {**self.model_kwargs, **generate_config}
        if stop:
            generate_config["stop"] = stop

        tools = self._choice_tools(tool_choice=self.tool_choice)
        response = await model.chat(
            messages=convert_to_openai_messages(messages),
            tools=tools,
            generate_config=generate_config,
        )

        async for stream_resp in response:
            if stream_resp:
                chunk = self._chat_response_to_chat_generation_chunk(stream_resp["choices"][0])
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    def _choice_tools(self, tool_choice: Optional[Union[str]] = None):
        """Select tools based on the tool_choice."""
        if tool_choice is None:
            return self.tools
        if isinstance(tool_choice, str):
            if tool_choice == "any":
                return self.tools
            if tool_choice == "none":
                return None
        elif isinstance(tool_choice, list):
            if len(tool_choice) == 0:
                return None
            else:
                return [tool for tool in self.tools if tool.name in tool_choice]
        elif tool_choice == Any:
            return self.tools
        else:
            raise ValueError("tool_choice must be None, a string or a list of strings.")

    def _build_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        built_tool_calls = []
        for tool_call in tool_calls:
            tc = dict()
            tc["id"] = tool_call["id"]
            tc["name"] = tool_call["function"]["name"]
            tc["args"] = tool_call["function"]["arguments"]
            built_tool_calls.append(tc)
        return built_tool_calls

    def _set_tools(self, tools, tool_choice):
        self.tools = tools
        self.tool_choice = tool_choice

    def bind_tools(
        self,
        tools: Sequence[
            Union[Dict[str, Any], type, Callable, BaseTool]  # noqa: UP006
        ],
        *,
        tool_choice: Optional[Union[str]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(convert_to_openai_tool(tool))
        model = deepcopy(self)
        model._set_tools(tool_choice=tool_choice, tools=formatted_tools)
        return model
