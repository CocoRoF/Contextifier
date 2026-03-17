# contextifier_new/services/__init__.py
"""
Services — Shared Format-Agnostic Processing Services

Services are stateful singleton-per-processor objects that provide
cross-cutting functionality shared by ALL handlers uniformly.

DEFINITION: A service belongs here if and only if:
1. It is FORMAT-AGNOSTIC — the same logic applies identically to
   every handler, regardless of source format.
2. It is SHARED — multiple handlers call the same service instance.
3. It provides a UTILITY — it transforms/renders data, it does NOT
   extract data from format-specific sources.

If logic is format-specific (e.g., extracting images from PDF pages
vs DOCX relationships), it belongs in the handler's ContentExtractor,
NOT in a service.

Service Catalog:
    TagService       — Create, detect, and remove structural tags
                       (page/slide/sheet/image/chart/metadata)
    ImageService     — Save image bytes, deduplicate, generate filenames
                       Delegates tag creation to TagService.
    ChartService     — Format ChartData into tagged text blocks
                       Delegates tag wrapping to TagService.
    TableService     — Format TableData into HTML/Markdown/Text
    MetadataService  — Format DocumentMetadata into tagged blocks
    StorageBackend   — Persist files (local/cloud)

Service Dependency Graph:
    TagService (standalone)
    ├─→ ImageService (depends on TagService + StorageBackend)
    ├─→ ChartService (depends on TagService)
    MetadataService (standalone)
    TableService (standalone)
    StorageBackend (standalone, used by ImageService)

Services are created once by DocumentProcessor and injected into all
handlers through their constructor. This ensures:
- Consistent tag formatting across all handlers
- Shared image dedup state within a processing session
- Single configuration source
"""

from contextifier.services.image_service import ImageService
from contextifier.services.tag_service import TagService
from contextifier.services.chart_service import ChartService
from contextifier.services.table_service import TableService
from contextifier.services.metadata_service import MetadataService

__all__ = [
    "ImageService",
    "TagService",
    "ChartService",
    "TableService",
    "MetadataService",
]
