from typing import TypedDict, Optional, List, Dict, Any

class PMSGState(TypedDict):
    question: str
    route: str  # 'retrieve' or 'forecast'
    retriever_mode: Optional[str]  # 'dense', 'bm25', or 'hybrid'
    retrieved_docs: List[Dict[str, Any]]
    forecast_result: Optional[Dict[str, Any]]
    visualization_data: List[Dict[str, Any]]
    answer: str
    answer_metadata: Dict[str, Any]
    citations: List[str]
    safety_check_passed: bool
    caveats: List[str]
