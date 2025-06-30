# python
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Union

import requests
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict


class XinferenceRerank(BaseDocumentCompressor):
    """Document compressor that uses `Xinference Rerank API`."""

    client: Any = None
    async_client: Any = None
    server_url: Optional[str] = None
    model_uid: Optional[str] = None
    top_n: Optional[int] = 3

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

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

        super().__init__(
            **{  # type: ignore[arg-type]
                "server_url": server_url,
                "model_uid": model_uid,
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

    def _rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        top_n: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
        return_documents: Optional[bool] = None,
        return_len: Optional[bool] = None,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        if not documents:
            items = []
        else:
            docs = [doc.page_content if isinstance(doc, Document) else doc for doc in documents]
            model = self.client.get_model(self.model_uid)
            top_n = top_n if (top_n is not None and top_n > 0) else self.top_n
            results = model.rerank(
                documents=docs,
                query=query,
                top_n=top_n,
                max_chunks_per_doc=max_chunks_per_doc,
                return_documents=return_documents,
                return_len=return_len,
                **kwargs,
            )
            if hasattr(results, "results"):
                results = results.results
            items = results["results"]
        for r in items:
            yield {"index": r["index"], "relevance_score": r["relevance_score"]}

    async def _arerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        top_n: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
        return_documents: Optional[bool] = None,
        return_len: Optional[bool] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        if not documents:
            items = []
        else:
            docs = [doc.page_content if isinstance(doc, Document) else doc for doc in documents]
            model = await self.async_client.get_model(self.model_uid)
            top_n = top_n if (top_n is not None and top_n > 0) else self.top_n
            results = await model.rerank(
                documents=docs,
                query=query,
                top_n=top_n,
                max_chunks_per_doc=max_chunks_per_doc,
                return_documents=return_documents,
                return_len=return_len,
                **kwargs,
            )
            if hasattr(results, "results"):
                results = results.results
            items = results["results"]
        for r in items:
            yield {"index": r["index"], "relevance_score": r["relevance_score"]}

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        from copy import deepcopy

        compressed = []
        for res in self._rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        from copy import deepcopy

        compressed = []
        async for res in self._arerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed
