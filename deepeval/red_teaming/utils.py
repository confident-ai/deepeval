from pydantic import BaseModel
from .schema import SyntheticData, SyntheticDataList

from deepeval.metrics.utils import trimAndLoadJson
from deepeval.models import DeepEvalBaseLLM


def generate_schema(
    prompt: str,
    schema: BaseModel,
    using_native_model: bool,
    model: DeepEvalBaseLLM,
) -> BaseModel:
    if using_native_model:
        res, _ = model.generate(prompt, schema=schema)
        return res
    else:
        try:
            res = model.generate(prompt, schema=schema)
            return res
        except TypeError:
            res = model.generate(prompt)
            data = trimAndLoadJson(res)
            if schema == SyntheticDataList:
                data_list = [SyntheticData(**item) for item in data["data"]]
                return SyntheticDataList(data=data_list)
            else:
                return schema(**data)


async def a_generate_schema(
    prompt: str,
    schema: BaseModel,
    using_native_model: bool,
    model: DeepEvalBaseLLM,
) -> BaseModel:
    if using_native_model:
        res, _ = await model.a_generate(prompt, schema=schema)
        return res
    else:
        try:
            res = await model.a_generate(prompt, schema=schema)
            return res
        except TypeError:
            res = await model.a_generate(prompt)
            data = trimAndLoadJson(res)
            if schema == SyntheticDataList:
                data_list = [SyntheticData(**item) for item in data["data"]]
                return SyntheticDataList(data=data_list)
            else:
                return schema(**data)
