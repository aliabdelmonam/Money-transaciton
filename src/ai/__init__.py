"""AI layer: OCR engines (adapters) + receipt extraction behind small protocols.

    from src.ai import AIConfig, ReceiptPipeline
    receipt = ReceiptPipeline(config=AIConfig(engine="paddle")).analyze_file("r.png").receipt
"""

from src.config.ai import AIConfig
from .extraction import Party, Receipt
from .pipeline import ReceiptAnalysis, ReceiptPipeline

__all__ = ["AIConfig", "Party", "Receipt", "ReceiptAnalysis", "ReceiptPipeline"]
