from .visdrone_eval import do_visdrone_evaluation


def visdrone_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
    filter_opt
):
    return do_visdrone_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        filter_opt=filter_opt
    )
