"""The agentic controller: validate, prune, route, execute, compose.

Lazy exports, like every other package here, so ``import satquery.agent`` stays
cheap and a missing optional dependency in one module cannot take down the rest.
"""

from .._lazy import lazy_getattr

_TABLE = {
    # the loop
    "Controller": ".controller:Controller",
    "QueryResult": ".controller:QueryResult",
    # workspace
    "Session": ".session:Session",
    "SessionStore": ".session:SessionStore",
    "SessionError": ".session:SessionError",
    "DEFAULT_SESSION": ".session:DEFAULT_SESSION",
    # validation
    "ImageReport": ".validate:ImageReport",
    "PairReport": ".validate:PairReport",
    "validate_image": ".validate:validate_image",
    "validate_pair": ".validate:validate_pair",
    # tools
    "TOOLS": ".tools:TOOLS",
    "ToolSpec": ".tools:ToolSpec",
    "ToolMenu": ".tools:ToolMenu",
    "ToolResult": ".tools:ToolResult",
    "ToolContext": ".tools:ToolContext",
    "build_menu": ".tools:build_menu",
    "run_tool": ".tools:run_tool",
    "scene_evidence": ".tools:scene_evidence",
    # routing
    "Plan": ".router:Plan",
    "PlannedCall": ".router:PlannedCall",
    "Router": ".router:Router",
    "RuleRouter": ".router:RuleRouter",
    "MistralRouter": ".router:MistralRouter",
    "RouterError": ".router:RouterError",
    "build_router": ".router:build_router",
    # composition
    "Composition": ".compose:Composition",
    "compose": ".compose:compose",
    # backends
    "backend_status": ".backends:backend_status",
    "select_vlm": ".backends:select_vlm",
    # trace
    "Trace": ".trace:Trace",
    "TraceEntry": ".trace:TraceEntry",
}

__all__ = sorted(_TABLE)
__getattr__ = lazy_getattr(__name__, _TABLE)
