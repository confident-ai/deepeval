import torch
from typing import Union, List, Optional
from typing import List, Union, get_origin
from deepeval.models.base_model import DeepEvalBaseModel
from deepeval.models._summac_model import _SummaCZS


class SummaCModels(DeepEvalBaseModel):
    def __init__(
        self,
        model_name: Optional[str] = None,
        granularity: Optional[str] = None,
        device: Optional[str] = None,
        *args,
        **kwargs
    ):
        model_name = "vitc" if model_name is None else model_name
        self.granularity = "sentence" if granularity is None else granularity
        self.device = (
            device
            if device is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        super().__init__(model_name, *args, **kwargs)

    def load_model(
        self,
        op1: Optional[str] = "max",
        op2: Optional[str] = "mean",
        use_ent: Optional[bool] = True,
        use_con: Optional[bool] = True,
        image_load_cache: Optional[bool] = True,
        **kwargs
    ):
        return _SummaCZS(
            model_name=self.model_name,
            granularity=self.granularity,
            device=self.device,
            op1=op1,
            op2=op2,
            use_con=use_con,
            use_ent=use_ent,
            imager_load_cache=image_load_cache,
            **kwargs
        )

    def _call(
        self, predictions: Union[str, List[str]], targets: Union[str, List[str]]
    ) -> Union[float, dict]:
        list_type = List[str]

        if (
            get_origin(predictions) is list_type
            and get_origin(targets) is list_type
        ):
            return self.model.score(targets, predictions)
        elif isinstance(predictions, str) and isinstance(targets, str):
            return self.model.score_one(targets, predictions)
        else:
            raise TypeError(
                "Either both predictions and targets should be List or both should be string"
            )
