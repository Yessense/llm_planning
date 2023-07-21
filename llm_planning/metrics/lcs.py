from typing import Any, Callable, Dict, List
from llm_planning.envs.base_env import BaseTask
from llm_planning.infrastructure.logger import BaseLogger
from llm_planning.metrics.base_metric import BaseMetric, BaseTaskMetrics, preprocess
from llm_planning.processors.base_processor import BaseProcessor


class LCS(BaseMetric):
    def __init__(self,
                 pred_process_f: Callable,
                 target_process_f: Callable,
                 name: str = 'LCS',
                 **kwargs):
        super().__init__(pred_process_f,
                         target_process_f,
                         name,
                         **kwargs)

    def __call__(self, pred: List, target: List) -> float:
        """Time Complexity: O(n*m)"""
        p = len(pred)
        t = len(target)

        # declaring the array for storing the dp values
        arr = [[None] * (t + 1) for _ in range(p + 1)]

        for i in range(p + 1):
            for j in range(t + 1):
                if i == 0 or j == 0:
                    arr[i][j] = 0
                elif pred[i - 1] == target[j - 1]:
                    arr[i][j] = arr[i - 1][j - 1] + 1
                else:
                    arr[i][j] = max(arr[i - 1][j], arr[i][j - 1])

        return arr[p][t] / max(p, t)


class LCSMetrics(BaseTaskMetrics):
    def __init__(self,
                 logger: BaseLogger,
                 processor: BaseProcessor,
                 **kwargs):
        super().__init__(logger, processor, **kwargs)
        # TODO: Fix this
        # Add metric functions
        to_list = list()
        a_lcs = LCS(to_list, to_list, 'A-LCS')
        p_lcs = LCS(to_list, to_list, 'P-LCS')
        pem = LCS(to_list, to_list, 'PEM')
        aem = LCS(to_list, to_list, 'AEM')
        self._metric_list: List = [a_lcs, p_lcs, pem, aem]
        # Dict to save intermediate values
        self._metric_values: Dict = dict()

    def update(self, predicted_task: BaseTask, target_task: BaseTask) -> Any:
        return super().update(predicted_task, target_task)

    def calculate_metrics(self) -> Any:
        return super().calculate_metrics()