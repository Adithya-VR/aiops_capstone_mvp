# AIOps Test Results

- Generated at: `2026-05-08T05:57:12+00:00`
- Total tests: `25`
- Passed: `25`
- Failed: `0`
- Errors: `0`
- Skipped: `0`
- Duration seconds: `7.92`

| Status | Test Case | Notes |
|---|---|---|
| PASS | `test_alert_generation.AlertGenerationTests.test_generate_alerts_includes_top_level` |  |
| PASS | `test_alert_generation.AlertGenerationTests.test_labeled_dataset_prefers_anomalous_representative_logs` |  |
| PASS | `test_alert_generation.AlertGenerationTests.test_severity_thresholds` |  |
| PASS | `test_api.APITests.test_log_search_and_cluster_filters_accept_user_input` |  |
| PASS | `test_api.APITests.test_missing_artifact_endpoint_uses_http_error` |  |
| PASS | `test_api.APITests.test_ready_dataset_endpoints` |  |
| PASS | `test_api.APITests.test_root_and_dataset_listing` |  |
| PASS | `test_api.APITests.test_unknown_dataset_returns_404` |  |
| PASS | `test_artifacts.ArtifactIntegrityTests.test_alert_artifact_schemas` |  |
| PASS | `test_artifacts.ArtifactIntegrityTests.test_feature_window_configuration` |  |
| PASS | `test_artifacts.ArtifactIntegrityTests.test_metrics_match_scores` |  |
| PASS | `test_config.DatasetConfigTests.test_available_datasets_shape` |  |
| PASS | `test_config.DatasetConfigTests.test_known_datasets_exist` |  |
| PASS | `test_config.DatasetConfigTests.test_paths_are_project_root_anchored` |  |
| PASS | `test_config.DatasetConfigTests.test_unknown_dataset_raises_key_error` |  |
| PASS | `test_model_outputs.ModelOutputTests.test_labeled_and_unlabeled_metric_modes_are_separate` |  |
| PASS | `test_model_outputs.ModelOutputTests.test_metrics_match_model_outputs` |  |
| PASS | `test_model_outputs.ModelOutputTests.test_openssh_uses_unlabeled_prediction_policy` |  |
| PASS | `test_model_outputs.ModelOutputTests.test_scores_are_valid_and_predictions_are_binary` |  |
| PASS | `test_model_outputs.ModelOutputTests.test_scores_have_required_model_columns` |  |
| PASS | `test_parsers.BGLParserTests.test_parses_empty_fatal_content` |  |
| PASS | `test_parsers.BGLParserTests.test_parses_labeled_failure_as_anomaly` |  |
| PASS | `test_parsers.OpenSSHParserTests.test_non_sshd_line_is_skipped` |  |
| PASS | `test_parsers.OpenSSHParserTests.test_parses_invalid_user_and_source_ip` |  |
| PASS | `test_parsers.OpenSSHParserTests.test_unrecognized_month_is_rejected` |  |
