from typing import Any, Callable, Dict, List
from llm_planning.datasets.base_dataset import BaseTask
from llm_planning.infrastructure.logger import BaseLogger
from llm_planning.metrics.base_metric import BaseMetric, BaseTaskMetrics, preprocess
from llm_planning.processors.base_processor import BaseProcessor


class LCSA(BaseMetric):
    """Longest common subarray"""

    def __init__(self,
                 pred_process_f: Callable,
                 target_process_f: Callable,
                 name: str = 'LCSA',
                 **kwargs):
        super().__init__(pred_process_f,
                         target_process_f,
                         name,
                         **kwargs)

    @preprocess
    def __call__(self, pred: List, target: List) -> float:
        p, t = len(pred), len(target)

        arr = [[0 for k in range(t+1)] for l in range(p+1)]
        mx = 0
        for i in range(p + 1):
            for j in range(t + 1):
                if (i == 0 or j == 0):
                    arr[i][j] = 0
                elif (pred[i-1] == target[j-1]):
                    arr[i][j] = arr[i-1][j-1] + 1
                    mx = max(mx, arr[i][j])
                else:
                    arr[i][j] = 0
        return mx / t


class LCSS(BaseMetric):
    """Longest common subsequence"""

    def __init__(self,
                 pred_process_f: Callable,
                 target_process_f: Callable,
                 name: str = 'LCSS',
                 normalize_both: bool = False,
                 **kwargs):
        super().__init__(pred_process_f,
                         target_process_f,
                         name,
                         **kwargs)
        self._normalize_both = normalize_both

    @preprocess
    def __call__(self, pred: List, target: List) -> float:
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
        
        if self._normalize_both:
            normalize_coef = max(p, t)
        else:
            normalize_coef = t

        return arr[p][t] / normalize_coef


class EM(LCSS):
    def __init__(self, pred_process_f: Callable[..., Any], target_process_f: Callable[..., Any], name: str = 'EM', normalize_both: bool = True, **kwargs):
        super().__init__(pred_process_f, target_process_f, name, normalize_both, **kwargs)

    def __call__(self, pred: List, target: List) -> float:
        return float(super().__call__(pred, target) == 1)


class LCSMetrics(BaseTaskMetrics):
    def __init__(self,
                 logger: BaseLogger,
                 processor: BaseProcessor,
                 **kwargs):
        super().__init__(logger, processor, **kwargs)
        # TODO: Fix this
        # Add metric functions
        # def to_text(task):
        #     return [task.text]

        def to_action_list(task):
            return [step.action for step in task.steps]

        def to_step_list(task):
            if len(task.steps):
                task.text = processor._steps_to_text(task.steps)

            task.steps = processor._text_to_steps(task.text)
            return [step.text for step in task.steps]

        a_lcsq = LCSS(to_action_list, to_action_list, 'A-LCSS')
        p_lcsq = LCSS(to_step_list, to_step_list, 'P-LCSS')
        a_lcsg = LCSA(to_action_list, to_action_list, 'A-LCSA')
        p_lcsg = LCSA(to_step_list, to_step_list, 'P-LCSA')
        pem = EM(to_step_list, to_step_list, 'PEM', normalize_both=True)
        aem = EM(to_action_list, to_action_list, 'AEM', normalize_both=True)
        psem = EM(to_step_list, to_step_list, 'PSEM', normalize_both=False)
        asem = EM(to_action_list, to_action_list, 'ASEM', normalize_both=False)
        self._metric_list: List[BaseMetric] = [
            a_lcsq, p_lcsq, a_lcsg, p_lcsg, pem, aem, psem, asem]
        # Dict to save intermediate values
        self._metric_values: Dict = {
            metric_cls.name: 0. for metric_cls in self._metric_list}
        self._count = 0.

    def update(self, predicted_task: BaseTask, target_task: BaseTask) -> str:
        self._count += 1

        metric_values: Dict = {}

        for metric_class in self._metric_list:
            metric_values[metric_class.name] = metric_class(
                predicted_task, target_task)

        for name, value in metric_values.items():
            self._metric_values[name] += value

        return metric_values
    
    def update_error(self):
        self._count += 1
        return "Error. Answer cannot be parsed"


    def calculate_metrics(self) -> Dict:
        total_metrics = {key: value / self._count for key,
                         value in self._metric_values.items()}
        return total_metrics
