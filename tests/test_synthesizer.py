import os
import pytest
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset


@pytest.mark.skip(reason="openai is expensive")
def test_synthesizer():
    module_b_dir = os.path.dirname(os.path.realpath(__file__))

    file_path = os.path.join(
        module_b_dir, "synthesizer_data", "pdf_example.pdf"
    )
    synthesizer = Synthesizer()
    synthesizer.generate_goldens_from_docs(
        document_paths=[file_path],
        include_expected_output=True,
        max_goldens_per_document=2,
    )
    synthesizer.save_as(file_type="json", directory="./results")


# module_b_dir = os.path.dirname(os.path.realpath(__file__))

# file_path = os.path.join(module_b_dir, "synthesizer_data", "pdf_example.pdf")
# synthesizer = Synthesizer(model="gpt-4")
# synthesizer.generate_goldens_from_docs(
#     synthesizer=synthesizer,
#     document_paths=[file_path],
#     max_goldens_per_document=2,
# )
# synthesizer.save_as(file_type="json", directory="./results")

# dataset = EvaluationDataset()
# dataset.generate_goldens_from_docs(
#     synthesizer=synthesizer,
#     document_paths=[file_path],
#     max_goldens_per_document=2,
# )
# dataset.save_as(file_type="json", directory="./results")


###########################################
######### Custom Text to SQL example ######
###########################################
json_customer_table = """{
      "name": "customers",
      "refSql": "select * from main.customers",
      "columns": [
        {
          "name": "City",
          "type": "VARCHAR",
          "isCalculated": false,
          "notNull": false,
          "properties": {
            "description": "The Customer City, where the customer company is located. Also called 'customer segment'."
          }
        },
        {
          "name": "Id",
          "type": "VARCHAR",
          "isCalculated": false,
          "notNull": false,
          "properties": {
            "description": "A unique identifier for each customer in the data model."
          }
        },
        {
          "name": "State",
          "type": "VARCHAR",
          "isCalculated": false,
          "notNull": false,
          "properties": {
            "description": "A field indicating the state where the customer is located."
          }
        },
        {
          "name": "orders",
          "type": "orders",
          "relationship": "CustomersOrders",
          "isCalculated": false,
          "notNull": false,
          "properties": {}
        },
        {
          "name": "LatestRecord",
          "type": "DATE",
          "isCalculated": true,
          "expression": "max(orders.PurchaseTimestamp)",
          "notNull": false,
          "properties": {}
        },
        {
            "name": "FirstRecord",
            "type": "DATE",
            "isCalculated": true,
            "expression": "min(orders.PurchaseTimestamp)",
            "notNull": false,
            "properties": {}
        },
        {
            "name": "VIP",
            "type": "BOOLEAN",
            "isCalculated": true,
            "expression": "sum(orders.Size) > 2",
            "notNull": false,
            "properties": {}
        },
        {
            "name": "OrderCount",
            "type": "BIGINT",
            "isCalculated": true,
            "expression": "count(orders.OrderId)",
            "notNull": false,
            "properties": {}
        },
        {
          "name": "Debit",
          "type": "DOUBLE",
          "isCalculated": true,
          "expression": "sum(orders.OrderBalance)",
          "notNull": false,
          "properties": {}
        },
        {
            "name": "ReviewRate",
            "type": "DOUBLE",
            "isCalculated": true,
            "expression": "count(orders.IsReviewed = TRUE) / count(DISTINCT orders.OrderId)",
            "notNull": false,
            "properties": {}
        }
      ],
      "primaryKey": "Id",
      "cached": false,
      "refreshTime": "30.00m",
      "properties": {
        "schema": "main",
        "catalog": "memory",
        "description": "A table of customers who have made purchases, including their city"
      }
    }
"""

schema = """
/* {"schema": "main", "catalog": "memory", "description": "A table of customers who have made purchases, including their city"} */
CREATE TABLE customers (
    -- {"description": "The Customer City, where the customer company is located. Also called \'customer segment\'."}
    City VARCHAR,
    -- {"description": "A unique identifier for each customer in the data model."}
    Id VARCHAR PRIMARY KEY,
    -- {"description": "A field indicating the state where the customer is located."}
    State VARCHAR,
    -- This column is a Calculated Field
    -- column expression: max(orders.PurchaseTimestamp)
    LatestRecord DATE,
    -- This column is a Calculated Field
    -- column expression: min(orders.PurchaseTimestamp)
    FirstRecord DATE,
    -- This column is a Calculated Field
    -- column expression: sum(orders.Size) > 2
    VIP BOOLEAN,
    -- This column is a Calculated Field
    -- column expression: count(orders.OrderId)
    OrderCount BIGINT,
    -- This column is a Calculated Field
    -- column expression: sum(orders.OrderBalance)
    Debit DOUBLE,
    -- This column is a Calculated Field
    -- column expression: count(orders.IsReviewed = TRUE) / count(DISTINCT orders.OrderId)
    ReviewRate DOUBLE
)
"""

# synthesizer = Synthesizer()
# synthesizer.generate_goldens(
#     contexts=[[json_customer_table, schema]],
#     max_goldens_per_context=5,
#     include_expected_output=True,
#     criteria=""
# )
# synthesizer.save_as(file_type="json", directory="./results")
