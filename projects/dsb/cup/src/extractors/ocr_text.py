"""
OCR text extraction using Surya OCR.
"""

from typing import override, final
from pdf2image import convert_from_path # pyright: ignore[reportUnknownVariableType]
from surya.recognition import RecognitionPredictor # pyright: ignore[reportMissingTypeStubs]
from surya.detection import DetectionPredictor # pyright: ignore[reportMissingTypeStubs]
from surya.layout import LayoutPredictor # pyright: ignore[reportMissingTypeStubs]
from surya.table_rec import TableRecPredictor # pyright: ignore[reportMissingTypeStubs]
from surya.recognition.schema import TextLine as SuryaTextLine # pyright: ignore[reportMissingTypeStubs]

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .base import BaseExtractor
from ..core.types import PageData, TextLine, TableData
from ..core.exceptions import PDFReadError

console = Console()


@final
class OCRTextExtractor(BaseExtractor):
  """Extract text from PDFs using Surya OCR."""
  
  def __init__(self) -> None:
    """Initialize the OCR text extractor."""
    super().__init__()
    console.print("🔄 Initializing Surya OCR models...", style="yellow")
    
    self.recognition_predictor: RecognitionPredictor = RecognitionPredictor()
    self.detection_predictor: DetectionPredictor = DetectionPredictor()
    self.layout_predictor: LayoutPredictor = LayoutPredictor()
    self.table_rec_predictor: TableRecPredictor = TableRecPredictor()
    
    console.print("✅ Surya OCR models initialized", style="green")
  
  @override
  def extract_text(self, pdf_path: str) -> list[PageData]:
    """Extract text from PDF using Surya OCR."""
    console.print(f"📄 Processing PDF: {pdf_path}", style="blue")
    
    # Convert PDF to images
    try:
      images = convert_from_path(pdf_path)
    except (OSError, IOError, ValueError) as e:
      raise PDFReadError(f"Error converting PDF to images: {e}") from e
    
    pages_data: list[PageData] = []
    
    with Progress(
      SpinnerColumn(),
      TextColumn("[progress.description]{task.description}"),
      console=console
    ) as progress:
      task = progress.add_task("Extracting text from pages...", total=len(images))
      
      for i, image in enumerate(images):
        progress.update(task, description=f"Processing page {i+1}/{len(images)}")
        
        # Get OCR results
        predictions = self.recognition_predictor(
          [image], 
          det_predictor=self.detection_predictor
        )
        
        # Handle the OCR result properly
        text_lines: list[TextLine] = []
        if predictions and len(predictions) > 0:
          prediction = predictions[0]
          # Check if prediction has text_lines attribute or is a dict
          if hasattr(prediction, "text_lines"):
            raw_lines = prediction.text_lines
          elif isinstance(prediction, dict) and "text_lines" in prediction:
            raw_lines: list[SuryaTextLine] = prediction["text_lines"] # pyright: ignore[reportUnknownVariableType]
          else:
            # Try to convert to dict if it's an OCRResult object
            try:
              raw_lines = prediction.text_lines if hasattr(prediction, "text_lines") else []
            except Exception:
              raw_lines = []
          
          # Convert to TextLine objects
          for line_idx, line_obj in enumerate(raw_lines, 1): # pyright: ignore[reportUnknownArgumentType]
            text = self._safe_get_text(line_obj)
            confidence = self._safe_get_confidence(line_obj)
            bbox = self._safe_get_bbox(line_obj)
            
            text_lines.append(TextLine(
              text=text,
              line_number=line_idx,
              page=i + 1,
              confidence=confidence,
              bbox=bbox
            ))
        
        page_data = PageData(
          page=i + 1,
          text_lines=text_lines,
          tables=[],  # Tables are handled separately
          total_lines=len(text_lines),
          raw_text="\n".join(line.text for line in text_lines)
        )
        pages_data.append(page_data)
        
        progress.advance(task)
    
    return pages_data
  
  @override
  def extract_tables(self, pdf_path: str) -> list[PageData]:
    """Extract tables from PDF using Surya table recognition."""
    console.print(f"📊 Extracting tables from PDF: {pdf_path}", style="blue")
    try:
      images = convert_from_path(pdf_path)
    except (OSError, IOError, ValueError) as e:
      raise PDFReadError(f"Error converting PDF to images: {e}") from e

    pages_data: list[PageData] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
      ) as progress:
      task = progress.add_task("Extracting tables from pages...", total=len(images))

      for i, image in enumerate(images):
        progress.update(task, description=f"Processing page {i+1}/{len(images)}")

        # Get table recognition results
        table_predictions = self.table_rec_predictor([image])

        page_tables: list[TableData] = []
        # Handle table predictions properly
        if table_predictions and len(table_predictions) > 0:
          page_tables_raw = table_predictions[0]
          if isinstance(page_tables_raw, list):
            for table_idx, table_data in enumerate(page_tables_raw): # pyright: ignore[reportUnknownVariableType]
              if isinstance(table_data, dict):
                table_data = table_data.copy() # pyright: ignore[reportUnknownVariableType]

                if cells := table_data.get("cells", []):
                  # Group cells by row
                  rows: dict[int, dict[int, str]] = {}
                  for cell in cells:
                    row_id = cell.get("row_id", 0)
                    col_id = cell.get("col_id", 0)
                    text = cell.get("text", "")
                    if row_id not in rows:
                      rows[row_id] = {}
                    rows[row_id][col_id] = text

                  # Convert to list of dictionaries
                  table_rows: list[dict[str, str]] = []
                  for row_id in sorted(rows.keys()):
                    row_dict: dict[str, str] = {}
                    for col_id in sorted(rows[row_id].keys()):
                      col_name = f"Column_{col_id}"
                      row_dict[col_name] = rows[row_id][col_id]
                    table_rows.append(row_dict)

                  # Determine columns
                  columns: list[str] = []
                  if table_rows:
                    columns = list(table_rows[0].keys())

                  table_result = TableData(
                    table_index=table_idx + 1,
                    page=i + 1,
                    rows=table_rows,
                    columns=columns,
                    shape=(len(table_rows), len(columns))
                  )
                  page_tables.append(table_result)

        page_data = PageData(
          page=i + 1,
          text_lines=[],
          tables=page_tables,
          total_lines=0,
          raw_text=""
        )
        pages_data.append(page_data)

        progress.advance(task)

    return pages_data
  
  def _safe_get_text[T](self, line_obj: T) -> str:
    """Safely extract text from a TextLine object or dict."""
    if hasattr(line_obj, "text"):
      return line_obj.text # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
    elif isinstance(line_obj, dict) and "text" in line_obj:
      return line_obj["text"] # pyright: ignore[reportUnknownVariableType]
    else:
      return str(line_obj) # pyright: ignore[reportUnknownArgumentType]
  
  def _safe_get_confidence[T](self, line_obj: T) -> float:
    """Safely extract confidence from a TextLine object or dict."""
    if hasattr(line_obj, "confidence"):
      return line_obj.confidence # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
    elif isinstance(line_obj, dict) and "confidence" in line_obj:
      return line_obj["confidence"] # pyright: ignore[reportUnknownVariableType]
    else:
      return 0.0
  
  def _safe_get_bbox[T](self, line_obj: T) -> tuple[float, float, float, float] | None:
    """Safely extract bounding box from a TextLine object or dict."""
    if hasattr(line_obj, "bbox"):
      return line_obj.bbox # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
    elif isinstance(line_obj, dict) and "bbox" in line_obj:
      return line_obj["bbox"] # pyright: ignore[reportUnknownVariableType]
    else:
      return None 