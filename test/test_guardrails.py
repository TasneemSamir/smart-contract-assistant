"""
Additional guardrail edge case tests.
"""

import pytest
from src.guardrails.safety import GuardRails


class TestGuardRailsEdgeCases:

    def setup_method(self):
        self.gr = GuardRails()

    def test_normal_contract_questions(self):
        """Common contract questions should all pass."""
        questions = [
            "What are the payment terms?",
            "Who are the parties in this agreement?",
            "What is the effective date?",
            "Can the contract be renewed?",
            "What are the liability limitations?",
            "Is there an arbitration clause?",
        ]
        for q in questions:
            is_safe, _ = self.gr.check_input(q)
            assert is_safe is True, f"Question wrongly blocked: {q}"

    def test_whitespace_only_blocked(self):
        """Whitespace-only input should be blocked."""
        is_safe, _ = self.gr.check_input("   \n\t  ")
        assert is_safe is False

    def test_good_output_passes(self):
        """Well-grounded output should pass with high confidence."""
        answer = "According to Section 3.1, the payment is due within 30 days."
        sources = [{"content": "Section 3.1 Payment terms: due within 30 days"}]
        processed, metadata = self.gr.check_output(answer, sources)
        assert metadata["confidence"] == "high"