"""Wrapper around Xinference embedding models."""

from typing import Any, Dict, List, Optional

import requests
from langchain_core.embeddings import Embeddings


class XinferenceEmbeddings(Embeddings):
    """Xinference embedding models.

    To use, you should have the xinference library installed:

    .. code-block:: bash

        pip install xinference

    If you're simply using the services provided by Xinference, you can utilize the xinference_client package:

    .. code-block:: bash

        pip install xinference_client

    Check out: https://github.com/xorbitsai/inference
    To run, you need to start a Xinference supervisor on one server and Xinference workers on the other servers.

    Example:
        To start a local instance of Xinference, run

        .. code-block:: bash

           $ xinference

        You can also deploy Xinference in a distributed cluster. Here are the steps:

        Starting the supervisor:

        .. code-block:: bash

           $ xinference-supervisor

        If you're simply using the services provided by Xinference, you can utilize the xinference_client package:

        .. code-block:: bash

            pip install xinference_client

        Starting the worker:

        .. code-block:: bash

           $ xinference-worker

    Then, launch a model using command line interface (CLI).

    Example:

    .. code-block:: bash

       $ xinference launch -n orca -s 3 -q q4_0

    It will return a model UID. Then you can use Xinference Embedding with LangChain.

    Example:

    .. code-block:: python

        from langchain_xinference import XinferenceEmbeddings

        xinference = XinferenceEmbeddings(
            server_url="http://0.0.0.0:9997",
            model_uid={model_uid},  # replace model_uid with the model UID return from launching the model
        )

    """  # noqa: E501

    client: Any
    async_client: Any
    server_url: Optional[str]
    """URL of the xinference server"""
    model_uid: Optional[str]
    """UID of the launched model"""

    def __init__(
        self,
        server_url: Optional[str] = None,
        model_uid: Optional[str] = None,
        api_key: Optional[str] = None,
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

        super().__init__()

        if server_url is None:
            raise ValueError("Please provide server URL")

        if model_uid is None:
            raise ValueError("Please provide the model UID")

        self.server_url = server_url

        self.model_uid = model_uid

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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Xinference.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """

        model = self.client.get_model(self.model_uid)

        embeddings = [model.create_embedding(text)["data"][0]["embedding"] for text in texts]
        return [list(map(float, e)) for e in embeddings]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Xinference.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """

        model = await self.async_client.get_model(self.model_uid)

        embeddings = [(await model.create_embedding(text))["data"][0]["embedding"] for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query of documents using Xinference.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """

        model = self.client.get_model(self.model_uid)

        embedding_res = model.create_embedding(text)

        embedding = embedding_res["data"][0]["embedding"]

        return list(map(float, embedding))

    async def aembed_query(self, text: str) -> List[float]:
        """Embed a query of documents using Xinference.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """

        model = await self.async_client.get_model(self.model_uid)

        embedding_res = await model.create_embedding(text)

        embedding = embedding_res["data"][0]["embedding"]

        return list(map(float, embedding))
