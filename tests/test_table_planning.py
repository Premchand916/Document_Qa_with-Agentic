"""Step 5 deterministic routing and validated table operations."""

import pandas as pd

from tests.support import OfflineCase


class TablePlanningTests(OfflineCase):
    def assets(self):
        return [
            {"source": "sales.csv", "sheet": "Data", "file_type": "csv",
             "dataframe": pd.DataFrame({"region": ["North", "South"], "amount": [10, 20]})},
            {"source": "second.csv", "sheet": "Data", "file_type": "csv",
             "dataframe": pd.DataFrame({"region": ["West"], "amount": [99]})},
        ]

    def test_pdf_question_uses_retrieval_with_csv_present(self):
        from agents.planner_agent import planner_agent
        state = planner_agent({"query": "What is the PDF cancellation policy?", "intent": "data_analysis",
                               "tabular_assets": self.assets(), "uploaded_file_types": ["pdf", "csv"]})
        self.assertFalse(state["use_tool"])
        self.assertEqual(state["document_type"], "document")

    def test_explicit_second_table_total_is_99(self):
        from agents.dataframe_tool import dataframe_tool_agent
        result = dataframe_tool_agent({"query": "Total amount in second.csv", "tabular_assets": self.assets()})
        self.assertEqual(result["tool_result"]["value"], 99)
        self.assertEqual(result["source"], "second.csv (Data)")
        self.assertIn("amount: 99", result["response"])

    def test_typed_north_filter_total_is_10(self):
        from agents.dataframe_tool import analyze_dataframe, dataframe_tool_agent
        asset = self.assets()[0]
        result = dataframe_tool_agent({"query": "Total amount for North only", "tabular_assets": [asset]})
        self.assertEqual(result["tool_result"]["value"], 10)
        self.assertEqual(result["tool_result"]["selected_rows"], [1])
        self.assertEqual(result["tool_result"]["filters"], [{"column": "region", "operator": "eq", "value": "North"}])
        self.assertEqual(analyze_dataframe(asset["dataframe"], "Total amount for North only")["total"]["amount"], 10)

    def test_nullable_numeric_sum_remains_40(self):
        from agents.dataframe_tool import dataframe_tool_agent
        frame = pd.DataFrame({"amount": pd.Series([10, None, 30], dtype="Int64")})
        result = dataframe_tool_agent({"query": "Total amount", "tabular_assets": [{"source": "missing.csv", "sheet": "Data",
                                                                                     "file_type": "csv", "dataframe": frame}]})
        self.assertEqual(result["tool_result"]["value"], 40)
        self.assertEqual(result["tool_result"]["selected_rows"], [1, 2, 3])

    def test_ambiguous_table_and_unknown_column_clarify(self):
        from agents.dataframe_tool import dataframe_tool_agent
        ambiguous = dataframe_tool_agent({"query": "Total amount", "tabular_assets": self.assets()})
        self.assertEqual(ambiguous["tool_plan"]["status"], "clarify")
        self.assertIn("sales.csv", ambiguous["response"])
        self.assertIn("second.csv", ambiguous["response"])
        unknown = dataframe_tool_agent({"query": "Total missing_column in sales.csv", "tabular_assets": self.assets()})
        self.assertEqual(unknown["tool_plan"]["status"], "clarify")
        self.assertIn("column", unknown["response"].lower())

    def test_unsupported_operation_and_forged_plan_cannot_run(self):
        from agents.dataframe_tool import dataframe_tool_agent
        unsupported = dataframe_tool_agent({"query": "Median amount in sales.csv", "tabular_assets": self.assets()})
        self.assertEqual(unsupported["tool_plan"]["status"], "clarify")
        self.assertIn("not supported", unsupported["response"])
        forged = dataframe_tool_agent({"query": "Total amount", "tabular_assets": self.assets(),
                                       "tool_plan": {"status": "ready", "asset_id": "outside.csv::Data",
                                                     "operation": "sum", "column": "amount", "filters": []}})
        self.assertEqual(forged["tool_plan"]["status"], "clarify")
        self.assertEqual(forged["documents"], [])

    def test_supported_operations_and_explicit_filter_have_typed_results(self):
        from agents.dataframe_tool import dataframe_tool_agent
        asset = self.assets()[0]
        cases = (("Average amount where region is North", "mean", 10),
                 ("Minimum amount", "min", 10),
                 ("Maximum amount", "max", 20),
                 ("Count amount", "count", 2),
                 ("How many rows", "row_count", 2))
        for query, operation, expected in cases:
            with self.subTest(query=query):
                result = dataframe_tool_agent({"query": query, "tabular_assets": [asset]})
                self.assertEqual(result["tool_result"]["operation"], operation)
                self.assertEqual(result["tool_result"]["value"], expected)
        schema = dataframe_tool_agent({"query": "Show columns", "tabular_assets": [asset]})
        self.assertEqual(schema["tool_result"]["value"]["columns"], ["region", "amount"])

    def test_explicit_sheet_selects_one_table(self):
        from agents.dataframe_tool import dataframe_tool_agent
        first, second = self.assets()
        first["source"] = second["source"] = "book.xlsx"
        first["sheet"], second["sheet"] = "January", "February"
        result = dataframe_tool_agent({"query": "Total amount in sheet February", "tabular_assets": [first, second]})
        self.assertEqual(result["tool_result"]["value"], 99)
        self.assertEqual(result["tool_result"]["sheet"], "February")
