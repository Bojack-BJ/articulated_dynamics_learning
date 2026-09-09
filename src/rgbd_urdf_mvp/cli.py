from __future__ import annotations

import json
import sys
import time

from .core.cli_config import YAMLSubsetError, expand_config_argv
from .core.serialization import load_episode, load_json, validate_episode
from .pipeline import RGBDToURDFPipeline
from .cli_parser import build_parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    In addition to normal subcommands, the CLI accepts a single YAML/JSON config
    file path or `--config config.yaml`, which expands into ordinary argv before
    argparse handles the command.
    """
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        expanded_argv = expand_config_argv(raw_argv, parser)
    except YAMLSubsetError as exc:
        parser.error(str(exc))
    args = parser.parse_args(expanded_argv)

    if args.command == "validate-episode":
        episode = load_episode(args.episode)
        errors = validate_episode(episode)
        if errors:
            for error in errors:
                print(error)
            return 1
        print(f"Episode '{episode.object_instance_id}' is valid for category '{episode.category}'.")
        return 0

    if args.command == "probe-torch-mps":
        from .core.torch_probe import probe_torch_mps

        print(
            json.dumps(
                probe_torch_mps(
                    run_tensor_test=not bool(args.no_tensor_test),
                    unsafe_force_mps=bool(args.unsafe_force_mps),
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "probe-mjx":
        from .core.mjx_probe import probe_mjx

        print(
            json.dumps(
                probe_mjx(
                    run_rollout_test=not bool(args.no_rollout_test),
                    jax_platform=str(args.jax_platform),
                    enable_pjrt_compatibility=args.enable_pjrt_compatibility,
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "run":
        episode = load_episode(args.episode)
        errors = validate_episode(episode)
        if errors:
            for error in errors:
                print(error)
            return 1
        result = RGBDToURDFPipeline().run(
            episode,
            args.output_dir,
            path_mode=str(args.path),
        )
        print(json.dumps({"path_mode": str(args.path), **result.to_dict()}, indent=2))
        return 0

    if args.command == "convert-gapartnet-mjcf":
        from .sim.gapartnet_adapter import GAPartNetMJCFAdapter, GAPartNetMJCFConfig

        output = GAPartNetMJCFAdapter().convert(
            GAPartNetMJCFConfig(
                asset_dir=args.asset_dir,
                output_mjcf=args.output_mjcf,
                urdf_name=str(args.urdf_name),
                density_kg_m3=float(args.density_kg_m3),
                minimum_proxy_size_m=float(args.minimum_proxy_size_m),
                add_floor=bool(args.add_floor),
                source_up_axis=str(args.source_up_axis),
            )
        )
        print(json.dumps({"gapartnet_mjcf": str(output.resolve())}, indent=2))
        return 0

    if args.command == "prepare-gapartnet-recordings":
        from .sim.gapartnet_dataset import (
            DEFAULT_PILOT_IDS,
            GAPartNetDatasetConfig,
            GAPartNetDatasetPreparer,
        )

        result = GAPartNetDatasetPreparer().prepare(
            GAPartNetDatasetConfig(
                archive_path=args.archive,
                output_dir=args.output_dir,
                object_ids=tuple(args.object_ids or DEFAULT_PILOT_IDS),
                force=bool(args.force),
                density_kg_m3=float(args.density_kg_m3),
            )
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "run-articulation-batch":
        from .batch.articulation_pipeline import ArticulationBatchConfig, ArticulationBatchRunner

        if args.jobs < 1:
            parser.error("--jobs must be a positive integer")
        result = ArticulationBatchRunner(
            ArticulationBatchConfig(
                manifest_path=args.manifest,
                resume=bool(args.resume),
                skip_existing=bool(args.skip_existing),
                jobs=int(args.jobs),
                output_root=args.output_root,
                batch_log_dir=args.batch_log_dir,
                torch_home=args.torch_home,
                record_config=args.record_config,
                track_config=args.track_config,
                cotracker_repo=args.cotracker_repo,
                cotracker_checkpoint=args.cotracker_checkpoint,
                track_device=args.track_device,
                tracking_jobs=args.tracking_jobs,
                stop_after=str(args.stop_after),
                fuse_pixel_stride=max(1, int(args.fuse_pixel_stride)),
                fuse_voxel_size_m=float(args.fuse_voxel_size_m),
                min_tracks_per_part=max(3, int(args.min_tracks_per_part)),
                joint_rotation_threshold_rad=args.joint_rotation_threshold_rad,
                joint_translation_threshold_m=args.joint_translation_threshold_m,
                mujoco_prior_mode=str(args.mujoco_prior),
                generate_viewer=not bool(args.no_generate_viewer),
                viewer_mode=str(args.viewer_mode),
                viewer_axis_remap=str(args.viewer_axis_remap),
                dynamics_backend=str(args.dynamics_backend),
                dynamics_config=args.dynamics_config,
                dynamics_jobs=args.dynamics_jobs,
                dynamics_jax_platform=args.dynamics_jax_platform,
                dynamics_enable_pjrt_compatibility=args.dynamics_enable_pjrt_compatibility,
                dynamics_render_gl_backend=args.dynamics_render_gl_backend,
                plot_dynamics=bool(args.plot_dynamics),
            )
        ).run()
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "convert-usd-mjcf-batch":
        from .batch.articulation_pipeline import UsdMjcfBatchConfig, UsdMjcfBatchConverter

        if args.jobs < 1:
            parser.error("--jobs must be a positive integer")
        result = UsdMjcfBatchConverter(
            UsdMjcfBatchConfig(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                converted_manifest=args.converted_manifest,
                jobs=int(args.jobs),
                force=bool(args.force),
            )
        ).run()
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "segment-episode-masks":
        from .perception.episode_masks import EpisodeMaskWriteConfig, EpisodeMaskWriter

        if args.frame_stride < 1:
            parser.error("--frame-stride must be a positive integer")
        if args.start_frame < 0:
            parser.error("--start-frame must be >= 0")
        if args.max_frames is not None and args.max_frames < 1:
            parser.error("--max-frames must be a positive integer when provided")
        episode_path = EpisodeMaskWriter().write(
            EpisodeMaskWriteConfig(
                episode_path=args.episode,
                output_episode_path=args.output_episode,
                output_dir=args.output_dir,
                provider=str(args.provider),
                mask_dir=args.mask_dir,
                command=args.provider_command,
                sam2_root=args.sam2_root,
                sam2_config=str(args.sam2_config),
                sam2_checkpoint=args.sam2_checkpoint,
                sam2_device=str(args.sam2_device),
                sam2_prompt_mode=str(args.sam2_prompt_mode),
                sam2_box_xyxy=args.sam2_box_xyxy,
                sam2_center_box_scale=float(args.sam2_center_box_scale),
                sam2_multimask=not bool(args.sam2_single_mask),
                sam2_mask_selection=str(args.sam2_mask_selection),
                sam2_mask_index=args.sam2_mask_index,
                sam3_checkpoint=args.sam3_checkpoint,
                sam3_device=str(args.sam3_device),
                sam3_prompt_mode=str(args.sam3_prompt_mode),
                sam3_box_xyxy=args.sam3_box_xyxy,
                sam3_center_box_scale=float(args.sam3_center_box_scale),
                sam3_mask_selection=str(args.sam3_mask_selection),
                sam3_mask_index=args.sam3_mask_index,
                sam3_conf=float(args.sam3_conf),
                mask_kind=str(args.mask_kind),
                frame_stride=int(args.frame_stride),
                start_frame=int(args.start_frame),
                max_frames=args.max_frames,
                view_indices=args.view_indices,
                threshold=int(args.threshold),
                force=bool(args.force),
            )
        )
        print(json.dumps({"episode_path": str(episode_path.resolve())}, indent=2))
        return 0

    if args.command == "segment-episode-masks-batch":
        from .perception.episode_masks import EpisodeMaskBatchConfig, EpisodeMaskBatchWriter

        if args.jobs < 1:
            parser.error("--jobs must be a positive integer")
        if args.frame_stride < 1:
            parser.error("--frame-stride must be a positive integer")
        if args.start_frame < 0:
            parser.error("--start-frame must be >= 0")
        if args.max_frames is not None and args.max_frames < 1:
            parser.error("--max-frames must be a positive integer when provided")
        result = EpisodeMaskBatchWriter(
            EpisodeMaskBatchConfig(
                manifest_path=args.manifest,
                jobs=int(args.jobs),
                provider=str(args.provider),
                output_root=args.output_root,
                sam2_root=args.sam2_root,
                sam2_config=str(args.sam2_config),
                sam2_checkpoint=args.sam2_checkpoint,
                sam2_device=str(args.sam2_device),
                sam2_prompt_mode=str(args.sam2_prompt_mode),
                sam2_box_xyxy=args.sam2_box_xyxy,
                sam2_center_box_scale=float(args.sam2_center_box_scale),
                sam2_multimask=not bool(args.sam2_single_mask),
                sam2_mask_selection=str(args.sam2_mask_selection),
                sam2_mask_index=args.sam2_mask_index,
                sam3_checkpoint=args.sam3_checkpoint,
                sam3_device=str(args.sam3_device),
                sam3_prompt_mode=str(args.sam3_prompt_mode),
                sam3_box_xyxy=args.sam3_box_xyxy,
                sam3_center_box_scale=float(args.sam3_center_box_scale),
                sam3_mask_selection=str(args.sam3_mask_selection),
                sam3_mask_index=args.sam3_mask_index,
                sam3_conf=float(args.sam3_conf),
                mask_kind=str(args.mask_kind),
                frame_stride=int(args.frame_stride),
                start_frame=int(args.start_frame),
                max_frames=args.max_frames,
                view_indices=args.view_indices,
                threshold=int(args.threshold),
                force=bool(args.force),
            )
        ).run()
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "propagate-episode-masks":
        from .perception.episode_masks import EpisodeMaskPropagationConfig, EpisodeMaskPropagator

        if args.reference_frame < 0:
            parser.error("--reference-frame must be >= 0")
        if args.start_frame < 0:
            parser.error("--start-frame must be >= 0")
        if args.end_frame is not None and args.end_frame < args.start_frame:
            parser.error("--end-frame must be >= --start-frame")
        if args.frame_stride < 1:
            parser.error("--frame-stride must be a positive integer")
        if args.seed_stride_px < 1:
            parser.error("--seed-stride-px must be a positive integer")
        if args.max_tracks_per_view < 1:
            parser.error("--max-tracks-per-view must be a positive integer")
        if args.mask_radius_px < 1:
            parser.error("--mask-radius-px must be a positive integer")
        episode_path = EpisodeMaskPropagator().propagate(
            EpisodeMaskPropagationConfig(
                episode_path=args.episode,
                output_episode_path=args.output_episode,
                output_dir=args.output_dir,
                backend=str(args.backend),
                mask_kind=str(args.mask_kind),
                sam2_root=args.sam2_root,
                sam2_config=str(args.sam2_config),
                sam2_checkpoint=args.sam2_checkpoint,
                sam2_device=str(args.sam2_device),
                sam2_part_mode=str(args.sam2_part_mode),
                sam2_offload_video_to_cpu=bool(args.sam2_offload_video_to_cpu),
                sam2_offload_state_to_cpu=bool(args.sam2_offload_state_to_cpu),
                reference_frame=int(args.reference_frame),
                start_frame=int(args.start_frame),
                end_frame=args.end_frame,
                frame_stride=int(args.frame_stride),
                view_indices=args.view_indices,
                device=str(args.device),
                cotracker_repo=args.cotracker_repo,
                cotracker_checkpoint=args.cotracker_checkpoint,
                cotracker_model=str(args.cotracker_model),
                seed_stride_px=int(args.seed_stride_px),
                max_tracks_per_view=int(args.max_tracks_per_view),
                visibility_threshold=float(args.visibility_threshold),
                mask_radius_px=int(args.mask_radius_px),
                threshold=int(args.threshold),
                force=bool(args.force),
            )
        )
        print(json.dumps({"episode_path": str(episode_path.resolve())}, indent=2))
        return 0

    if args.command == "evaluate-episode-masks":
        from .perception.episode_masks import EpisodeMaskEvaluationConfig, EpisodeMaskEvaluator

        output_json = EpisodeMaskEvaluator().evaluate(
            EpisodeMaskEvaluationConfig(
                predicted_episode_path=args.predicted_episode,
                reference_episode_path=args.reference_episode,
                output_json=args.output_json,
                mask_kind=str(args.mask_kind),
                threshold=int(args.threshold),
            )
        )
        print(json.dumps({"mask_evaluation": str(output_json.resolve())}, indent=2))
        return 0

    if args.command == "fuse-pointcloud":
        from .perception.pointcloud_fusion import EpisodePointCloudFuser, PointCloudFusionConfig

        manifest_path = EpisodePointCloudFuser(
            PointCloudFusionConfig(
                episode_path=args.episode,
                output_dir=args.output_dir,
                pixel_stride=max(1, int(args.pixel_stride)),
                voxel_size_m=float(args.voxel_size_m),
                bbox_margin_m=float(args.bbox_margin_m),
                near_depth_band_m=float(args.near_depth_band_m),
                allow_shared_pose_fallback=not bool(args.no_shared_pose_fallback),
            )
        ).fuse()
        print(json.dumps({"manifest_path": str(manifest_path.resolve())}, indent=2))
        return 0

    if args.command == "visualize-pointcloud":
        from .perception.pointcloud_viz import PointCloudViewerBuilder, PointCloudVisualizationConfig

        html_path = PointCloudViewerBuilder(
            PointCloudVisualizationConfig(
                input_path=args.input,
                output_html=args.output_html,
                max_points_per_frame=max(1, int(args.max_points_per_frame)),
                point_radius_px=float(args.point_radius_px),
                part_pose_path=args.part_poses_json,
                part_track_path=args.part_tracks_json,
                joint_inference_path=args.joint_inference_json,
            )
        ).build()
        print(json.dumps({"viewer_html": str(html_path.resolve())}, indent=2))
        return 0

    if args.command == "visualize-object-mask-diagnostics":
        from .perception.object_mask_diagnostics_viz import (
            ObjectMaskDiagnosticsVisualizationConfig,
            ObjectMaskDiagnosticsVisualizer,
        )

        manifest_path = ObjectMaskDiagnosticsVisualizer().build(
            ObjectMaskDiagnosticsVisualizationConfig(
                motion_tracks=args.motion_tracks,
                output_dir=args.output_dir,
                joint_inference=args.joint_inference,
                gt_joint_annotation=args.gt_joint_annotation,
                evaluation_json=args.evaluation_json,
                local_split_summary=args.local_split_summary,
                candidate_eval=args.candidate_eval,
                frame=args.frame,
                top_n_candidates=max(1, int(args.top_n_candidates)),
                axis_length_m=float(args.axis_length_m),
                viz_frame=args.viz_frame,
                axis_remap=args.axis_remap,
                flow_min_motion_m=float(args.flow_min_motion),
                flow_max_tracks=max(1, int(args.flow_max_tracks)),
                flow_subsample=max(1, int(args.flow_subsample)),
                flow_scale=float(args.flow_scale),
                make_matplotlib=bool(args.make_matplotlib),
                plot_projections=args.plot_projections,
                make_animation=bool(args.make_animation),
                animation_fps=max(1, int(args.animation_fps)),
                animation_max_tracks=max(1, int(args.animation_max_tracks)),
                animation_color_by=str(args.animation_color_by),
            )
        )
        print(json.dumps({"object_mask_diagnostics_viz": str(manifest_path.resolve())}, indent=2))
        return 0

    if args.command == "visualize-object-mask-flow-html":
        from .perception.object_mask_flow_html import ObjectMaskFlowHtmlBuilder, ObjectMaskFlowHtmlConfig

        html_path = ObjectMaskFlowHtmlBuilder().build(
            ObjectMaskFlowHtmlConfig(
                motion_tracks=args.motion_tracks,
                output_html=args.output_html,
                joint_inference=args.joint_inference,
                gt_joint_annotation=args.gt_joint_annotation,
                evaluation_json=args.evaluation_json,
                background_fusion_manifest=args.background_fusion_manifest,
                background_episode=args.background_episode,
                background_exclude_object_mask=bool(args.background_exclude_object_mask),
                background_max_points=max(1, int(args.background_max_points)),
                background_persistent=bool(args.background_persistent),
                background_voxel_size_m=float(args.background_voxel_size_m),
                mjcf_replay_episode=args.mjcf_replay_episode,
                mjcf_mesh_opacity=float(args.mjcf_mesh_opacity),
                max_tracks=max(1, int(args.max_tracks)),
                frame_stride=max(1, int(args.frame_stride)),
                full_timeline=bool(args.full_timeline),
                trail_length=max(0, int(args.trail_length)),
                axis_remap=args.axis_remap,
                color_by=str(args.color_by),
            )
        )
        print(json.dumps({"object_mask_flow_html": str(html_path.resolve())}, indent=2))
        return 0

    if args.command == "compute-track-quality":
        from .perception.track_quality import TrackQualityAnalyzer, TrackQualityConfig

        outputs = TrackQualityAnalyzer().analyze(
            TrackQualityConfig(
                input_tracks=args.motion_tracks,
                output_dir=args.output_dir,
                bad_track_threshold=float(args.bad_track_threshold),
                bad_timestep_threshold=float(args.bad_timestep_threshold),
                mask_bad_timesteps=bool(args.mask_bad_timesteps),
                max_step_m=(None if args.max_step_m is None else max(0.0, float(args.max_step_m))),
                keep_query_connected_segment=bool(args.keep_query_connected_segment),
                min_query_connected_frames=max(1, int(args.min_query_connected_frames)),
                query_connected_max_gap_frames=max(
                    0, int(args.query_connected_max_gap_frames)
                ),
                spatial_dbscan_eps_m=(
                    None
                    if args.spatial_dbscan_eps_m is None
                    else max(1e-9, float(args.spatial_dbscan_eps_m))
                ),
                spatial_dbscan_min_samples=max(2, int(args.spatial_dbscan_min_samples)),
            )
        )
        print(json.dumps({key: str(path.resolve()) for key, path in outputs.items()}, indent=2))
        return 0

    if args.command == "track-part-pixels":
        from .perception.part_tracking import PartPixelTracker, PartPixelTrackingConfig

        output_json = PartPixelTracker(
            PartPixelTrackingConfig(
                episode_path=args.episode,
                output_json=args.output_json,
                device=str(args.device),
                cotracker_repo=args.cotracker_repo,
                cotracker_checkpoint=args.cotracker_checkpoint,
                cotracker_model=str(args.cotracker_model),
                unsafe_force_mps=bool(args.unsafe_force_mps),
                reference_frame=int(args.reference_frame),
                frame_stride=max(1, int(args.frame_stride)),
                seed_stride_px=max(1, int(args.seed_stride_px)),
                max_tracks_per_part_view=max(1, int(args.max_tracks_per_part_view)),
                max_queries_per_forward=max(1, int(args.max_queries_per_forward)),
                visibility_threshold=float(args.visibility_threshold),
                depth_consistency_window_radius_px=max(0, int(args.depth_consistency_window_radius_px)),
                depth_consistency_max_delta_m=max(0.0, float(args.depth_consistency_max_delta_m)),
                repair_temporal_depth_spikes=bool(args.repair_temporal_depth_spikes),
                depth_spike_jump_threshold_m=max(0.0, float(args.depth_spike_jump_threshold_m)),
                depth_spike_neighbor_tolerance_m=max(0.0, float(args.depth_spike_neighbor_tolerance_m)),
                depth_spike_max_run_frames=max(1, int(args.depth_spike_max_run_frames)),
                require_part_mask_consistency=not bool(args.no_part_mask_consistency),
                strict_object_mask_consistency=bool(args.strict_object_mask_consistency),
                allow_backward_tracking=not bool(args.no_backward_tracking),
                show_progress=not bool(args.no_progress),
                export_cotracker_features=bool(args.export_cotracker_features),
                cotracker_features_output=args.cotracker_features_output,
                dynamic_reseeding=bool(args.dynamic_reseeding),
                dynamic_reseed_bidirectional=bool(args.dynamic_reseed_bidirectional),
                reseed_interval_frames=max(1, int(args.reseed_interval_frames)),
                reseed_coverage_radius_px=max(0.0, float(args.reseed_coverage_radius_px)),
                reseed_bbox_scale=max(1.0, float(args.reseed_bbox_scale)),
                reseed_max_tracks_per_frame_view=max(1, int(args.reseed_max_tracks_per_frame_view)),
                reseed_max_tracks_per_view=max(1, int(args.reseed_max_tracks_per_view)),
            )
        ).track()
        print(json.dumps({"part_tracks_artifact": str(output_json.resolve())}, indent=2))
        return 0

    if args.command in {"probe-cotracker-features", "probe-track-features"}:
        from .perception.cotracker_features import CoTrackerFeatureProbeConfig, CoTrackerFeatureProber

        output_json = CoTrackerFeatureProber(
            CoTrackerFeatureProbeConfig(
                tracks_path=args.tracks,
                features_npz=args.features,
                output_json=args.output_json,
                output_embedding_csv=args.output_embedding_csv,
                label_field=str(args.label_field),
                max_pairs=max(1, int(args.max_pairs)),
                cluster_k=args.cluster_k,
                seed=int(args.seed),
            )
        ).probe()
        print(json.dumps({"track_feature_probe": str(output_json.resolve())}, indent=2))
        return 0

    if args.command == "prepare-tapip3d-input":
        from .perception.tapip3d_adapter import TAPIP3DInputConfig, TAPIP3DInputPreparer

        output_path = TAPIP3DInputPreparer(
            TAPIP3DInputConfig(
                episode_path=args.episode,
                output_npz=args.output_npz,
                view_index=int(args.view_index),
                frame_stride=max(1, int(args.frame_stride)),
                seed_tracks=args.seed_tracks,
            )
        ).prepare()
        print(json.dumps({"tapip3d_input": str(output_path)}, indent=2))
        return 0

    if args.command == "import-tapip3d-tracks":
        from .perception.tapip3d_adapter import TAPIP3DImportConfig, TAPIP3DTrackImporter

        output_path = TAPIP3DTrackImporter(
            TAPIP3DImportConfig(
                tapip_input_npz=args.tapip_input,
                tapip_result_npz=args.tapip_result,
                seed_tracks=args.seed_tracks,
                output_json=args.output_json,
                visibility_threshold=float(args.visibility_threshold),
            )
        ).import_tracks()
        print(json.dumps({"tapip3d_tracks": str(output_path)}, indent=2))
        return 0

    if args.command == "merge-tapip3d-views":
        from .perception.tapip3d_adapter import TAPIP3DMergeConfig, TAPIP3DMultiViewMerger

        tracks_path, features_path = TAPIP3DMultiViewMerger(
            TAPIP3DMergeConfig(
                track_paths=list(args.tracks),
                output_tracks=args.output_tracks,
                feature_paths=None if args.features is None else list(args.features),
                output_features=args.output_features,
            )
        ).merge()
        print(json.dumps({
            "tapip3d_tracks": str(tracks_path),
            "tapip3d_features": None if features_path is None else str(features_path),
        }, indent=2))
        return 0

    if args.command == "train-track-slot-head":
        from .perception.track_slot_head import TrackSlotTrainer, TrackSlotTrainingConfig

        model_path = TrackSlotTrainer(TrackSlotTrainingConfig(
            manifest_path=args.manifest, output_dir=args.output_dir,
            max_slots=max(2, int(args.max_slots)), hidden_dim=max(16, int(args.hidden_dim)),
            epochs=max(1, int(args.epochs)), learning_rate=max(1e-8, float(args.learning_rate)),
            weight_decay=max(0.0, float(args.weight_decay)),
            pairwise_loss_weight=max(0.0, float(args.pairwise_loss_weight)),
            device=str(args.device), seed=int(args.seed),
        )).train()
        print(json.dumps({"track_slot_model": str(model_path)}, indent=2))
        return 0

    if args.command == "predict-track-slots":
        from .perception.track_slot_head import TrackSlotPredictionConfig, TrackSlotPredictor

        output_path, viewer_path = TrackSlotPredictor(TrackSlotPredictionConfig(
            tracks_path=args.tracks, features_npz=args.features, model_path=args.model,
            output_json=args.output_json, output_viewer=args.output_viewer,
            device=str(args.device), generate_viewer=not bool(args.no_viewer),
        )).predict()
        print(json.dumps({
            "track_slot_predictions": str(output_path),
            "viewer": None if viewer_path is None else str(viewer_path),
        }, indent=2))
        return 0

    if args.command == "train-pairwise-affinity":
        from .perception.pairwise_affinity import PairwiseAffinityTrainer, PairwiseAffinityTrainingConfig

        model_path = PairwiseAffinityTrainer(
            PairwiseAffinityTrainingConfig(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                knn_k=max(1, int(args.knn_k)),
                min_common_frames=max(2, int(args.min_common_frames)),
                epochs=max(1, int(args.epochs)),
                batch_size=max(1, int(args.batch_size)),
                learning_rate=max(1e-8, float(args.learning_rate)),
                weight_decay=max(0.0, float(args.weight_decay)),
                hidden_dim=max(16, int(args.hidden_dim)),
                hard_negative_weight=max(0.0, float(args.hard_negative_weight)),
                hard_mining_k=max(0, int(args.hard_mining_k)),
                max_positive_negative_ratio=max(0.0, float(args.max_positive_negative_ratio)),
                device=str(args.device),
                seed=int(args.seed),
            )
        ).train()
        print(json.dumps({"pairwise_affinity_model": str(model_path.resolve())}, indent=2))
        return 0

    if args.command == "evaluate-pairwise-affinity":
        from .perception.pairwise_affinity import PairwiseAffinityEvaluationConfig, PairwiseAffinityEvaluator

        output_json = PairwiseAffinityEvaluator(
            PairwiseAffinityEvaluationConfig(
                manifest_path=args.manifest,
                model_path=args.model,
                output_json=args.output_json,
                split=str(args.split),
                knn_k=max(1, int(args.knn_k)),
                min_common_frames=max(2, int(args.min_common_frames)),
                device=str(args.device),
            )
        ).evaluate()
        print(json.dumps({"pairwise_affinity_evaluation": str(output_json.resolve())}, indent=2))
        return 0

    if args.command == "train-motion-part-slots":
        from .perception.motion_part_slots import MotionPartSlotTrainer, MotionPartSlotTrainingConfig

        model_path = MotionPartSlotTrainer(
            MotionPartSlotTrainingConfig(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                max_slots=max(2, int(args.max_slots)),
                hidden_dim=max(16, int(args.hidden_dim)),
                encoder_layers=max(1, int(args.encoder_layers)),
                decoder_layers=max(1, int(args.decoder_layers)),
                attention_heads=max(1, int(args.attention_heads)),
                epochs=max(1, int(args.epochs)),
                learning_rate=max(1e-8, float(args.learning_rate)),
                weight_decay=max(0.0, float(args.weight_decay)),
                rigid_loss_weight=max(0.0, float(args.rigid_loss_weight)),
                dice_loss_weight=max(0.0, float(args.dice_loss_weight)),
                pairwise_loss_weight=max(0.0, float(args.pairwise_loss_weight)),
                existence_loss_weight=max(0.0, float(args.existence_loss_weight)),
                part_balanced_assignment=not bool(args.no_part_balanced_assignment),
                canonicalize_geometry=not bool(args.no_canonicalize_geometry),
                geometry_augmentation=not bool(args.no_geometry_augmentation),
                geometry_noise_std=max(0.0, float(args.geometry_noise_std)),
                depth_bias_probability=max(0.0, min(float(args.depth_bias_probability), 1.0)),
                depth_bias_std=max(0.0, float(args.depth_bias_std)),
                sampled_depth_spike_probability=max(
                    0.0, min(float(args.sampled_depth_spike_probability), 1.0)
                ),
                track_dropout_ratio=max(0.0, min(float(args.track_dropout_ratio), 0.8)),
                view_dropout_probability=max(
                    0.0, min(float(args.view_dropout_probability), 1.0)
                ),
                max_dropped_views=max(0, int(args.max_dropped_views)),
                pair_samples_per_object=max(0, int(args.pair_samples_per_object)),
                object_batch_size=max(1, int(args.object_batch_size)),
                data_loader_workers=max(0, int(args.data_loader_workers)),
                topology_balanced_sampling=not bool(args.no_topology_balanced_sampling),
                collapse_fixed_connected_labels=bool(args.collapse_fixed_connected_labels),
                device=str(args.device),
                seed=int(args.seed),
            )
        ).train()
        print(json.dumps({"motion_part_slot_model": str(model_path.resolve())}, indent=2))
        return 0

    if args.command == "infer-motion-part-slots":
        from .perception.motion_part_slots import MotionPartSlotInferenceConfig, MotionPartSlotInferencer

        output_json = MotionPartSlotInferencer(
            MotionPartSlotInferenceConfig(
                tracks_path=args.tracks,
                features_npz=args.features_npz,
                model_path=args.model,
                output_json=args.output_json,
                device=str(args.device),
                post_ransac_refine=bool(args.post_ransac_refine),
                ransac_iterations=max(1, int(args.ransac_iterations)),
                ransac_inlier_threshold_m=max(1e-6, float(args.ransac_inlier_threshold_m)),
                ransac_min_inliers=max(3, int(args.ransac_min_inliers)),
                slot_existence_threshold=max(0.0, min(float(args.slot_existence_threshold), 1.0)),
                min_visible_frames=max(0, int(args.min_visible_frames)),
                min_visible_ratio=max(0.0, min(float(args.min_visible_ratio), 1.0)),
                max_trajectory_jump_m=max(0.0, float(args.max_trajectory_jump_m)),
                seed=int(args.seed),
            )
        ).infer()
        print(json.dumps({"motion_part_tracks": str(output_json.resolve())}, indent=2))
        return 0

    if args.command == "train-slot-relation-head":
        from .kinematics.pairwise_relation_head import SlotRelationTrainer, SlotRelationTrainingConfig

        model_path = SlotRelationTrainer(
            SlotRelationTrainingConfig(
                manifest_path=args.manifest,
                slot_model_path=args.slot_model,
                output_dir=args.output_dir,
                epochs=max(1, int(args.epochs)),
                object_batch_size=max(1, int(args.object_batch_size)),
                hidden_dim=max(16, int(args.hidden_dim)),
                learning_rate=max(1e-8, float(args.learning_rate)),
                weight_decay=max(0.0, float(args.weight_decay)),
                edge_positive_weight=max(1.0, float(args.edge_positive_weight)),
                joint_type_loss_weight=max(0.0, float(args.joint_type_loss_weight)),
                axis_loss_weight=max(0.0, float(args.axis_loss_weight)),
                axis_line_loss_weight=max(0.0, float(args.axis_line_loss_weight)),
                joint_replay_loss_weight=max(0.0, float(args.joint_replay_loss_weight)),
                slot_assignment_loss_weight=max(0.0, float(args.slot_assignment_loss_weight)),
                slot_dice_loss_weight=max(0.0, float(args.slot_dice_loss_weight)),
                slot_pairwise_loss_weight=max(0.0, float(args.slot_pairwise_loss_weight)),
                slot_pair_samples_per_object=max(0, int(args.slot_pair_samples_per_object)),
                slot_rigid_loss_weight=max(0.0, float(args.slot_rigid_loss_weight)),
                slot_existence_loss_weight=max(0.0, float(args.slot_existence_loss_weight)),
                slot_assignment_consistency_loss_weight=max(
                    0.0, float(args.slot_assignment_consistency_loss_weight)
                ),
                joint_type_balanced_loss=not bool(args.no_joint_type_balanced_loss),
                rotation_augmentation=(
                    bool(args.rotation_augmentation) and not bool(args.no_rotation_augmentation)
                ),
                rotation_augmentation_probability=float(args.rotation_augmentation_probability),
                rotation_augmentation_mode=str(args.rotation_augmentation_mode),
                rotation_augmentation_scope=str(args.rotation_augmentation_scope),
                slot_input_mode=str(args.slot_input_mode),
                slot_geometry_representation=str(args.slot_geometry_representation),
                temporal_occlusion_augmentation=bool(args.temporal_occlusion_augmentation),
                temporal_occlusion_probability=max(
                    0.0, min(float(args.temporal_occlusion_probability), 1.0)
                ),
                temporal_occlusion_min_fraction=max(
                    0.0, min(float(args.temporal_occlusion_min_fraction), 0.9)
                ),
                temporal_occlusion_max_fraction=max(
                    0.0, min(float(args.temporal_occlusion_max_fraction), 0.9)
                ),
                temporal_occlusion_track_fraction=max(
                    0.0, min(float(args.temporal_occlusion_track_fraction), 1.0)
                ),
                trajectory_corruption_augmentation=bool(args.trajectory_corruption_augmentation),
                trajectory_corruption_probability=max(
                    0.0, min(float(args.trajectory_corruption_probability), 1.0)
                ),
                trajectory_corruption_track_fraction=max(
                    0.0, min(float(args.trajectory_corruption_track_fraction), 1.0)
                ),
                trajectory_drift_scale_fraction=max(
                    0.0, float(args.trajectory_drift_scale_fraction)
                ),
                trajectory_spike_probability=max(
                    0.0, min(float(args.trajectory_spike_probability), 1.0)
                ),
                coherent_drift_probability=max(
                    0.0, min(float(args.coherent_drift_probability), 1.0)
                ),
                recovery_offset_probability=max(
                    0.0, min(float(args.recovery_offset_probability), 1.0)
                ),
                track_id_switch_probability=max(
                    0.0, min(float(args.track_id_switch_probability), 1.0)
                ),
                slot_contamination_probability=max(
                    0.0, min(float(args.slot_contamination_probability), 1.0)
                ),
                slot_contamination_fraction=max(
                    0.0, min(float(args.slot_contamination_fraction), 1.0)
                ),
                excitation_weighted_axis_loss=bool(args.excitation_weighted_axis_loss),
                min_axis_excitation=max(0.0, float(args.min_axis_excitation)),
                full_axis_excitation=max(0.0, float(args.full_axis_excitation)),
                hard_axis_focal_gamma=max(0.0, float(args.hard_axis_focal_gamma)),
                hard_axis_max_weight=max(1.0, float(args.hard_axis_max_weight)),
                axis_geometry_branch=bool(args.axis_geometry_branch),
                axis_head_type=str(args.axis_head_type),
                vector_pivot_parameterization=str(args.vector_pivot_parameterization),
                relation_train_scope=str(args.relation_train_scope),
                geometry_encoder_type=str(args.geometry_encoder_type),
                trajectory_hidden_dim=max(16, int(args.trajectory_hidden_dim)),
                trajectory_samples=max(2, int(args.trajectory_samples)),
                quality_weighted_trajectories=bool(args.quality_weighted_trajectories),
                robust_segment_weights=bool(args.robust_segment_weights),
                geometry_max_tracks=max(1, int(args.geometry_max_tracks)),
                geometry_attention_heads=max(1, int(args.geometry_attention_heads)),
                geometry_transformer_layers=max(1, int(args.geometry_transformer_layers)),
                axis_equivariance_loss_weight=max(0.0, float(args.axis_equivariance_loss_weight)),
                axis_line_equivariance_loss_weight=max(
                    0.0, float(args.axis_line_equivariance_loss_weight)
                ),
                edge_consistency_loss_weight=max(0.0, float(args.edge_consistency_loss_weight)),
                type_consistency_loss_weight=max(0.0, float(args.type_consistency_loss_weight)),
                unfreeze_slot_backbone=bool(args.unfreeze_slot_backbone),
                slot_unfreeze_scope=str(args.slot_unfreeze_scope),
                slot_learning_rate_scale=max(0.0, float(args.slot_learning_rate_scale)),
                initial_relation_model_path=args.initial_relation_model,
                load_initial_slot_state=not bool(args.ignore_initial_slot_state),
                device=str(args.device),
                seed=int(args.seed),
            )
        ).train()
        print(json.dumps({"slot_relation_model": str(model_path.resolve())}, indent=2))
        return 0

    if args.command == "infer-slot-relation-head":
        from .kinematics.pairwise_relation_head import SlotRelationInferencer, SlotRelationInferenceConfig

        output_json = SlotRelationInferencer(
            SlotRelationInferenceConfig(
                tracks_path=args.tracks,
                features_npz=args.features_npz,
                slot_model_path=args.slot_model,
                relation_model_path=args.relation_model,
                output_json=args.output_json,
                device=str(args.device),
                slot_existence_threshold=max(0.0, min(1.0, float(args.slot_existence_threshold))),
                edge_threshold=max(0.0, min(1.0, float(args.edge_threshold))),
                min_axis_confidence=max(0.0, float(args.min_axis_confidence)),
                min_axis_observability=max(0.0, float(args.min_axis_observability)),
                min_edge_observability=max(0.0, float(args.min_edge_observability)),
                gate_selected_edges=bool(args.gate_selected_edges),
                quality_weighted_trajectories=args.quality_weighted_trajectories,
                min_trajectory_quality=max(0.0, float(args.min_trajectory_quality)),
                robust_segment_weights=args.robust_segment_weights,
                trajectory_assignment_override=args.trajectory_assignment_override,
            )
        ).infer()
        print(json.dumps({"slot_relation_predictions": str(output_json.resolve())}, indent=2))
        return 0

    if args.command == "estimate-part-poses":
        method = str(args.method)
        if method == "auto" and args.input.suffix.lower() == ".json":
            payload = load_json(args.input)
            method = "tracks" if isinstance(payload.get("tracks"), list) else "pca"
        elif method == "auto":
            method = "pca"

        if method == "tracks":
            from .perception.part_tracking import TrackPartPoseEstimationConfig, TrackPartPoseEstimator

            output_json = TrackPartPoseEstimator(
                TrackPartPoseEstimationConfig(
                    input_path=args.input,
                    output_json=args.output_json,
                    min_tracks_per_part=max(3, int(args.min_tracks_per_part)),
                    anchor_part_id=args.anchor_part_id,
                    anchor_selection=str(args.anchor_selection),
                    quality_weighted=bool(args.quality_weighted),
                    quality_weight_field=str(args.quality_weight_field),
                    track_quality_field=str(args.track_quality_field),
                    min_timestep_weight=float(args.min_timestep_weight),
                )
            ).estimate()
        else:
            from .perception.part_pose import PartPoseEstimationConfig, PartPoseEstimator

            output_json = PartPoseEstimator(
                PartPoseEstimationConfig(
                    input_path=args.input,
                    output_json=args.output_json,
                    min_points_per_part=max(1, int(args.min_points_per_part)),
                    anchor_part_id=args.anchor_part_id,
                )
            ).estimate()
        print(json.dumps({"part_pose_artifact": str(output_json.resolve())}, indent=2))
        return 0

    if args.command == "segment-motion-parts":
        from .perception.motion_segmentation import MotionPartSegmentationConfig, MotionPartSegmenter

        output_json = MotionPartSegmenter(
            MotionPartSegmentationConfig(
                input_tracks=args.input_tracks,
                output_json=args.output_json,
                mode=args.mode,
                diagnostics_json=args.diagnostics_json,
                sweep_output_dir=args.sweep_output_dir,
                rigidity_threshold_m=float(args.rigidity_threshold_m),
                max_neighbor_distance_m=float(args.max_neighbor_distance_m),
                min_common_frames=max(2, int(args.min_common_frames)),
                min_tracks_per_part=max(1, int(args.min_tracks_per_part)),
                min_motion_m=float(args.min_motion_m),
                static_motion_threshold_m=float(args.static_motion_threshold_m),
                moving_motion_threshold_m=float(args.moving_motion_threshold_m),
                quality_filter=bool(args.quality_filter),
                min_visible_frames=max(2, int(args.min_visible_frames)),
                max_depth_jump_m=float(args.max_depth_jump_m),
                max_trajectory_jump_m=float(args.max_trajectory_jump_m),
                knn_k=max(1, int(args.knn_k)),
                k_min=max(2, int(args.k_min)),
                k_max=max(2, int(args.k_max)),
                spectral_k=args.spectral_k,
                edge_ablation=args.edge_ablation,
                quality_weighted_affinity=bool(args.quality_weighted_affinity),
                quality_weighted_affinity_time_only=bool(args.quality_weighted_affinity_time_only),
                quality_weighted_affinity_edge_prior=bool(args.quality_weighted_affinity_edge_prior),
                quality_affinity_min_pair_weight=max(0.0, min(1.0, float(args.quality_affinity_min_pair_weight))),
                articulation_compatible_affinity=bool(args.articulation_compatible_affinity),
                articulation_type_mismatch_penalty=max(
                    0.0,
                    min(1.0, float(args.articulation_type_mismatch_penalty)),
                ),
                articulation_static_mismatch_penalty=max(
                    0.0,
                    min(1.0, float(args.articulation_static_mismatch_penalty)),
                ),
                articulation_min_motion_for_type_penalty=max(
                    0.0,
                    float(args.articulation_min_motion_for_type_penalty),
                ),
                articulation_min_confidence_for_type_penalty=max(
                    0.0,
                    min(1.0, float(args.articulation_min_confidence_for_type_penalty)),
                ),
                cotracker_features_npz=args.cotracker_features_npz,
                learned_affinity_floor=max(0.0, min(1.0, float(args.learned_affinity_floor))),
                pairwise_affinity_model=args.pairwise_affinity_model,
                pairwise_affinity_floor=max(0.0, min(1.0, float(args.pairwise_affinity_floor))),
                pairwise_affinity_device=str(args.pairwise_affinity_device),
                pairwise_connect_threshold=max(0.0, min(1.0, float(args.pairwise_connect_threshold))),
                skip_base_bridge_checks=bool(args.skip_base_bridge_checks),
                ransac_iterations=max(1, int(args.ransac_iterations)),
                ransac_sample_size=max(3, int(args.ransac_sample_size)),
                ransac_inlier_threshold_m=max(1e-6, float(args.ransac_inlier_threshold_m)),
                ransac_min_inliers=max(3, int(args.ransac_min_inliers)),
                ransac_max_models=max(1, int(args.ransac_max_models)),
                ransac_spatial_link_m=max(1e-6, float(args.ransac_spatial_link_m)),
                ransac_assignment_threshold_m=max(1e-6, float(args.ransac_assignment_threshold_m)),
                ransac_seed=int(args.ransac_seed),
                ransac_base_stabilize=bool(args.ransac_base_stabilize),
                ransac_learned_seed=bool(args.ransac_learned_seed),
            )
        ).segment()
        print(json.dumps({"motion_part_tracks": str(output_json.resolve())}, indent=2))
        return 0

    if args.command == "local-split-motion-cluster":
        from .perception.motion_segmentation import LocalMotionClusterSplitter, LocalSplitDiagnosticsConfig

        summary_json = LocalMotionClusterSplitter(
            LocalSplitDiagnosticsConfig(
                input_tracks=args.input_tracks,
                evaluation_json=args.evaluation_json,
                output_dir=args.output_dir,
                split_cluster_ids=args.split_cluster_id,
                local_k_min=max(2, int(args.local_k_min)),
                local_k_max=max(2, int(args.local_k_max)),
                knn_k=max(1, int(args.knn_k)),
                edge_ablation=args.edge_ablation,
                rigidity_threshold_m=float(args.rigidity_threshold_m),
                max_neighbor_distance_m=float(args.max_neighbor_distance_m),
                min_common_frames=max(2, int(args.min_common_frames)),
            )
        ).split()
        print(json.dumps({"local_split_summary": str(summary_json.resolve())}, indent=2))
        return 0

    if args.command == "infer-joints":
        from .kinematics.joint_inference import JointInferenceConfig, JointInferencer

        output_json = JointInferencer(
            JointInferenceConfig(
                input_path=args.input,
                output_json=args.output_json,
                rotation_threshold_rad=float(args.rotation_threshold_rad),
                translation_threshold_m=float(args.translation_threshold_m),
                use_track_translation_axis=not bool(args.no_track_translation_axis),
                use_track_residual_type=not bool(args.no_track_residual_type),
                track_residual_requires_pose_candidate=bool(args.track_residual_requires_pose_candidate),
                track_residual_decision_ratio=float(args.track_residual_decision_ratio),
                min_track_residual_samples=max(1, int(args.min_track_residual_samples)),
                min_track_residual_tracks=max(1, int(args.min_track_residual_tracks)),
                robust_track_model_trim_ratio=max(0.0, min(0.8, float(args.robust_track_model_trim_ratio))),
                quality_weighted_replay=bool(args.quality_weighted_replay),
                min_replay_weight=max(0.0, min(1.0, float(args.min_replay_weight))),
                mujoco_prior=args.mujoco_prior,
                orient_parent_by_motion=bool(args.orient_parent_by_motion),
                parent_orientation_motion_margin_m=float(args.parent_orientation_motion_margin_m),
            )
        ).infer()
        print(json.dumps({"joint_inference_artifact": str(output_json.resolve())}, indent=2))
        return 0

    if args.command == "export-inferred-articulation":
        from .kinematics.inferred_articulation import InferredArticulationPipeline, InferredArticulationPipelineConfig

        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else args.joint_inference.resolve().parent / "inferred_articulation"
        )
        result = InferredArticulationPipeline().run(
            InferredArticulationPipelineConfig(
                episode_path=args.episode,
                part_pose_path=args.part_poses,
                joint_inference_path=args.joint_inference,
                output_dir=output_dir,
            )
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "identify-dynamics":
        if args.render_gl_backend not in {"auto", "none"}:
            import os

            os.environ["MUJOCO_GL"] = str(args.render_gl_backend)
        from .dynamics.system_id import DynamicsIdentificationConfig, DynamicsIdentifier

        artifact_path = DynamicsIdentifier().run(
            DynamicsIdentificationConfig(
                episode_path=args.episode,
                articulation_artifact_path=args.articulation_artifact,
                mjcf_path=args.mjcf,
                output_dir=args.output_dir,
                max_iterations=max(1, int(args.max_iterations)),
                learning_rate=float(args.learning_rate),
                q_weight=float(args.q_weight),
                qdot_weight=float(args.qdot_weight),
                prior_weight=float(args.prior_weight),
                optimize_static_parts=bool(args.optimize_static_parts),
                optimize_mass=not bool(args.no_optimize_mass),
                optimize_damping=not bool(args.no_optimize_damping),
                optimize_friction=not bool(args.no_optimize_friction),
                enable_contact=bool(args.enable_contact),
                render_gl_backend=str(args.render_gl_backend),
                gravity_mode=str(args.gravity_mode),
            )
        )
        print(json.dumps({"dynamics_identification_artifact": str(artifact_path.resolve())}, indent=2))
        return 0

    if args.command == "identify-dynamics-mjx":
        if args.render_gl_backend not in {"auto", "none"}:
            import os

            os.environ["MUJOCO_GL"] = str(args.render_gl_backend)
        from .core.jax_runtime import configure_jax_runtime
        from .dynamics.system_id_mjx import MJXDynamicsIdentificationConfig, MJXDynamicsIdentifier

        configure_jax_runtime(
            platform=str(args.jax_platform),
            enable_pjrt_compatibility=args.enable_pjrt_compatibility,
        )
        artifact_path = MJXDynamicsIdentifier().run(
            MJXDynamicsIdentificationConfig(
                episode_path=args.episode,
                articulation_artifact_path=args.articulation_artifact,
                mjcf_path=args.mjcf,
                output_dir=args.output_dir,
                max_iterations=max(1, int(args.max_iterations)),
                learning_rate=float(args.learning_rate),
                q_weight=float(args.q_weight),
                qdot_weight=float(args.qdot_weight),
                prior_weight=float(args.prior_weight),
                optimize_static_parts=bool(args.optimize_static_parts),
                optimize_mass=not bool(args.no_optimize_mass),
                optimize_damping=not bool(args.no_optimize_damping),
                optimize_friction=not bool(args.no_optimize_friction),
                enable_contact=bool(args.enable_contact),
                jit=not bool(args.no_jit),
                jax_platform=str(args.jax_platform),
                enable_pjrt_compatibility=args.enable_pjrt_compatibility,
                render_gl_backend=str(args.render_gl_backend),
            )
        )
        print(json.dumps({"dynamics_identification_artifact": str(artifact_path.resolve())}, indent=2))
        return 0

    if args.command == "plot-dynamics-identification":
        from .dynamics.plotting import DynamicsOptimizationPlotConfig, DynamicsOptimizationPlotter

        output_svg = DynamicsOptimizationPlotter().plot(
            DynamicsOptimizationPlotConfig(
                input_path=args.input,
                output_svg=args.output_svg,
                log_loss=not bool(args.linear_loss),
                width=max(720, int(args.width)),
                height=max(480, int(args.height)),
            )
        )
        print(json.dumps({"optimization_history_svg": str(output_svg.resolve())}, indent=2))
        return 0

    if args.command == "plan-manipulation":
        from .manipulation.mujoco_push_planner import MuJoCoPushPlanConfig, MuJoCoPushPlanner

        artifact_path = MuJoCoPushPlanner().run(
            MuJoCoPushPlanConfig(
                mjcf_path=args.mjcf,
                target_q=float(args.target_q),
                output_dir=args.output_dir,
                dynamics_identification_path=args.dynamics_identification,
                joint_name=args.joint_name,
                joint_id=args.joint_id,
                mode=str(args.mode),
                initial_q=args.initial_q,
                duration_s=float(args.duration_s),
                sim_dt=args.sim_dt,
                num_candidates=max(3, int(args.num_candidates)),
                max_initial_qvel=float(args.max_initial_qvel),
                max_pulse_force=float(args.max_pulse_force),
                pulse_duration_s=float(args.pulse_duration_s),
                tolerance=float(args.tolerance),
                enable_contact=bool(args.enable_contact),
                gravity_mode=str(args.gravity_mode),
            )
        )
        print(json.dumps({"manipulation_plan": str(artifact_path.resolve())}, indent=2))
        return 0

    if args.command == "generate-il-demos":
        from .manipulation.il_dataset import BallisticILDatasetConfig, BallisticILDatasetGenerator

        dataset_path = BallisticILDatasetGenerator().generate(
            BallisticILDatasetConfig(
                mjcf_path=args.mjcf,
                output_dir=args.output_dir,
                task=args.task,
                num_episodes=max(1, int(args.num_episodes)),
                release_duration_s=float(args.release_duration_s),
                condition_source=args.condition_source,
                joint_name=args.joint_name,
                joint_id=args.joint_id,
                sim_dt=args.sim_dt,
                max_initial_qvel=float(args.max_initial_qvel),
                num_teacher_candidates=max(3, int(args.num_teacher_candidates)),
                response_samples=max(2, int(args.response_samples)),
                tolerance=float(args.tolerance),
                qdot_tolerance=float(args.qdot_tolerance),
                seed=int(args.seed),
                enable_contact=bool(args.enable_contact),
                gravity_mode=str(args.gravity_mode),
            )
        )
        print(json.dumps({"il_dataset": str(dataset_path.resolve())}, indent=2))
        return 0

    if args.command == "train-il-policy":
        from .manipulation.il_train import BallisticILTrainConfig, BallisticILTrainer

        policy_path = BallisticILTrainer().train(
            BallisticILTrainConfig(
                dataset_path=args.dataset,
                output_dir=args.output_dir,
                condition_mode=args.condition_mode,
                epochs=max(1, int(args.epochs)),
                batch_size=max(1, int(args.batch_size)),
                learning_rate=float(args.learning_rate),
                hidden_dim=max(1, int(args.hidden_dim)),
                hidden_layers=max(1, int(args.hidden_layers)),
                val_fraction=float(args.val_fraction),
                seed=int(args.seed),
                noisy_condition_std=float(args.noisy_condition_std),
            )
        )
        print(json.dumps({"il_policy": str(policy_path.resolve())}, indent=2))
        return 0

    if args.command == "eval-il-policy":
        from .manipulation.il_eval import BallisticILEvalConfig, BallisticILEvaluator

        eval_path = BallisticILEvaluator().evaluate(
            BallisticILEvalConfig(
                policy_path=args.policy,
                mjcf_path=args.mjcf,
                output_dir=args.output_dir,
                num_episodes=max(1, int(args.num_episodes)),
                release_duration_s=float(args.release_duration_s),
                joint_name=args.joint_name,
                joint_id=args.joint_id,
                sim_dt=args.sim_dt,
                response_samples=max(2, int(args.response_samples)),
                tolerance=float(args.tolerance),
                qdot_tolerance=float(args.qdot_tolerance),
                seed=int(args.seed),
                max_abs_initial_qvel=float(args.max_abs_initial_qvel),
                enable_contact=bool(args.enable_contact),
                gravity_mode=str(args.gravity_mode),
            )
        )
        print(json.dumps({"il_eval": str(eval_path.resolve())}, indent=2))
        return 0

    if args.command == "prepare-arx-x5-model":
        from .manipulation.arx_x5 import ARXX5ModelPreparer, ARXX5PrepareConfig

        manifest_path = ARXX5ModelPreparer().prepare(
            ARXX5PrepareConfig(
                output_dir=args.output_dir,
                source_dir=args.source_dir,
                repo_url=str(args.repo_url),
                validate_mujoco=not bool(args.no_validate_mujoco),
            )
        )
        print(json.dumps({"arx_x5_model_manifest": str(manifest_path.resolve())}, indent=2))
        return 0

    if args.command == "generate-contact-il-demos":
        from .manipulation.contact_il import ContactILDatasetConfig, ContactILDatasetGenerator

        dataset_path = ContactILDatasetGenerator().generate(
            ContactILDatasetConfig(
                mjcf_path=args.mjcf,
                output_dir=args.output_dir,
                robot_model_path=args.robot_model,
                robot_end_effector_body=str(args.robot_end_effector_body),
                task=args.task,
                num_episodes=max(1, int(args.num_episodes)),
                release_duration_s=float(args.release_duration_s),
                contact_duration_s=float(args.contact_duration_s),
                condition_source=args.condition_source,
                joint_name=args.joint_name,
                joint_id=args.joint_id,
                sim_dt=args.sim_dt,
                max_force=float(args.max_force),
                num_teacher_candidates=max(3, int(args.num_teacher_candidates)),
                response_samples=max(2, int(args.response_samples)),
                tolerance=float(args.tolerance),
                qdot_tolerance=float(args.qdot_tolerance),
                seed=int(args.seed),
                enable_contact=bool(args.enable_contact),
                gravity_mode=str(args.gravity_mode),
            )
        )
        print(json.dumps({"contact_il_dataset": str(dataset_path.resolve())}, indent=2))
        return 0

    if args.command == "eval-contact-il-policy":
        from .manipulation.contact_il import ContactILEvalConfig, ContactILEvaluator

        eval_path = ContactILEvaluator().evaluate(
            ContactILEvalConfig(
                policy_path=args.policy,
                mjcf_path=args.mjcf,
                output_dir=args.output_dir,
                num_episodes=max(1, int(args.num_episodes)),
                release_duration_s=float(args.release_duration_s),
                joint_name=args.joint_name,
                joint_id=args.joint_id,
                sim_dt=args.sim_dt,
                response_samples=max(2, int(args.response_samples)),
                tolerance=float(args.tolerance),
                qdot_tolerance=float(args.qdot_tolerance),
                seed=int(args.seed),
                max_force=float(args.max_force),
                max_contact_duration_s=float(args.max_contact_duration_s),
                enable_contact=bool(args.enable_contact),
                gravity_mode=str(args.gravity_mode),
            )
        )
        print(json.dumps({"contact_il_eval": str(eval_path.resolve())}, indent=2))
        return 0

    if args.command == "visualize-il-dataset":
        from .manipulation.il_viz import ILDatasetVisualizationConfig, ILDatasetVisualizer

        svg_path = ILDatasetVisualizer().visualize(
            ILDatasetVisualizationConfig(
                dataset_path=args.dataset,
                output_svg=args.output_svg,
                max_trajectories=max(1, int(args.max_trajectories)),
                title=args.title,
            )
        )
        print(json.dumps({"il_dataset_svg": str(svg_path.resolve())}, indent=2))
        return 0

    if args.command == "render-il-rollout":
        from .manipulation.il_animation import ILRolloutRenderConfig, ILRolloutRenderer

        render_path = ILRolloutRenderer().render(
            ILRolloutRenderConfig(
                dataset_path=args.dataset,
                mjcf_path=args.mjcf,
                output_path=args.output_path,
                robot_model_path=args.robot_model,
                dynamics_artifact_path=args.dynamics_artifact,
                episode_index=int(args.episode_index),
                trajectory_source=str(args.trajectory_source),
                joint_name=args.joint_name,
                joint_id=args.joint_id,
                sim_dt=args.sim_dt,
                width=max(64, int(args.width)),
                height=max(64, int(args.height)),
                fps=max(1, int(args.fps)),
                camera_name=args.camera_name,
                camera_azimuth=float(args.camera_azimuth),
                camera_elevation=float(args.camera_elevation),
                camera_distance=args.camera_distance,
                object_visual_scale=float(args.object_visual_scale),
                playback_slowdown=float(args.playback_slowdown),
                robot_base_pos=tuple(float(value) for value in args.robot_base_pos),
                robot_base_euler=tuple(float(value) for value in args.robot_base_euler),
                robot_motion=str(args.robot_motion),
                robot_ee_body=str(args.robot_ee_body),
                robot_approach_distance=float(args.robot_approach_distance),
                robot_pulse_distance=float(args.robot_pulse_distance),
                dynamics_body_name=args.dynamics_body_name,
                dynamics_joint_name=args.dynamics_joint_name,
                enable_contact=bool(args.enable_contact),
                avoid_robot_self_collision=not bool(args.allow_robot_self_collision),
                self_collision_penetration_tolerance_m=float(args.self_collision_penetration_tolerance_m),
                avoid_robot_object_collision=not bool(args.allow_robot_object_collision),
                object_collision_penetration_tolerance_m=float(args.object_collision_penetration_tolerance_m),
                gravity_mode=str(args.gravity_mode),
                write_frames=bool(args.write_frames),
            )
        )
        print(json.dumps({"il_rollout_animation": str(render_path.resolve())}, indent=2))
        return 0

    if args.command == "prepare-generation-images":
        from .perception.generation_preprocess import GenerationImagePreparationConfig, GenerationImagePreparer

        manifest_path = GenerationImagePreparer().prepare(
            GenerationImagePreparationConfig(
                episode_path=args.episode,
                output_dir=args.output_dir,
                frame_index=int(args.frame_index),
                view_indices=args.view_indices,
                image_views=args.image_views,
                mask_source=str(args.mask_source),
                background=str(args.background),
                padding_ratio=float(args.padding_ratio),
                min_mask_pixels=max(1, int(args.min_mask_pixels)),
            )
        )
        print(json.dumps({"generation_image_manifest": str(manifest_path.resolve())}, indent=2))
        return 0

    if args.command == "hunyuan3d-generate":
        from .perception.hunyuan3d_client import Hunyuan3DClient, Hunyuan3DGenerationConfig
        from .perception.generation_preprocess import load_generation_image_manifest

        image_path = args.image
        image_views = args.image_views
        if args.image_manifest is not None:
            image_path, image_views = load_generation_image_manifest(args.image_manifest)

        output_path = Hunyuan3DClient(
            server_url=args.server_url,
            api_token=args.api_token,
            timeout_s=min(float(args.timeout_s), 120.0),
        ).generate(
            Hunyuan3DGenerationConfig(
                server_url=args.server_url,
                output_path=args.output,
                image_path=image_path,
                image_views=image_views,
                text=args.text,
                mesh_path=args.mesh,
                mode=args.mode,
                texture=bool(args.texture),
                seed=int(args.seed),
                output_type=args.type,
                octree_resolution=args.octree_resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                face_count=args.face_count,
                timeout_s=float(args.timeout_s),
                poll_interval_s=float(args.poll_interval_s),
                api_token=args.api_token,
                download_result=not bool(args.remote_skip_download),
            )
        )
        print(json.dumps({"generated_model_path": str(output_path.resolve())}, indent=2))
        return 0

    if args.command == "particulate-infer":
        from .perception.particulate_adapter import ParticulateInferenceConfig, ParticulateInferenceRunner

        manifest_path = ParticulateInferenceRunner().run(
            ParticulateInferenceConfig(
                mesh_path=args.mesh,
                reconstruction_artifact_path=args.reconstruction_artifact,
                output_dir=args.output_dir,
                particulate_root=args.particulate_root,
                python_bin=args.python_bin,
                model_config=args.model_config,
                ckpt_path=args.ckpt_path,
                up_dir=args.up_dir,
                num_points=max(1, int(args.num_points)),
                num_points_global=max(1, int(args.num_points_global)),
                target_faces=args.target_faces,
                min_part_confidence=float(args.min_part_confidence),
                strict=not bool(args.no_strict),
                animation_frames=max(2, int(args.animation_frames)),
                export_urdf=not bool(args.no_export_urdf),
                export_mjcf=not bool(args.no_export_mjcf),
                eval=not bool(args.no_eval),
                dry_run=bool(args.dry_run),
            )
        )
        print(json.dumps({"particulate_result": str(manifest_path.resolve())}, indent=2))
        return 0

    if args.command == "remote-articulate-generate":
        from .perception.generation_preprocess import (
            GenerationImagePreparationConfig,
            GenerationImagePreparer,
            load_generation_image_manifest,
        )
        from .perception.remote_articulation_client import RemoteArticulationClient, RemoteArticulationConfig

        prepare_started = time.perf_counter()
        prepare_timing_s: float | None = None
        image_path = args.image
        image_views = args.image_views
        generation_image_manifest = args.image_manifest
        if args.episode is not None:
            generation_image_manifest = GenerationImagePreparer().prepare(
                GenerationImagePreparationConfig(
                    episode_path=args.episode,
                    output_dir=args.generation_image_output_dir,
                    frame_index=int(args.frame_index),
                    view_indices=args.view_indices,
                    image_views=args.image_views,
                    mask_source=args.mask_source,
                    background=args.background,
                    padding_ratio=float(args.padding_ratio),
                    min_mask_pixels=max(1, int(args.min_mask_pixels)),
                )
            )
            prepare_timing_s = time.perf_counter() - prepare_started
        if generation_image_manifest is not None:
            image_path, image_views = load_generation_image_manifest(generation_image_manifest)

        output_dir = RemoteArticulationClient(
            server_url=args.server_url,
            api_token=args.api_token,
            timeout_s=min(float(args.timeout_s), 120.0),
        ).generate(
            RemoteArticulationConfig(
                server_url=args.server_url,
                output_dir=args.output_dir,
                image_path=image_path,
                image_views=image_views,
                text=args.text,
                texture=bool(args.texture),
                seed=int(args.seed),
                output_type=args.type,
                octree_resolution=args.octree_resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                face_count=args.face_count,
                particulate_up_dir=args.particulate_up_dir,
                particulate_num_points=max(1, int(args.particulate_num_points)),
                particulate_global_points=max(1, int(args.particulate_global_points)),
                particulate_target_faces=args.particulate_target_faces,
                particulate_min_part_confidence=float(args.particulate_min_part_confidence),
                particulate_strict=not bool(args.particulate_no_strict),
                timeout_s=float(args.timeout_s),
                poll_interval_s=float(args.poll_interval_s),
                api_token=args.api_token,
                download_result=not bool(args.remote_skip_download),
            )
        )
        timing_path = output_dir / "remote_articulation_timing.json"
        if prepare_timing_s is not None and timing_path.exists():
            timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
            client_timings = timing_payload.setdefault("client_timings", {})
            if isinstance(client_timings, dict):
                client_timings["prepare_generation_images_s"] = prepare_timing_s
            timing_path.write_text(json.dumps(timing_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"remote_articulation_output_dir": str(output_dir.resolve())}, indent=2))
        return 0

    if args.command == "export-reart-sequence":
        from .perception.reart_adapter import ReArtSequenceExportConfig, ReArtSequenceExporter

        output_dir = ReArtSequenceExporter().export(
            ReArtSequenceExportConfig(
                fusion_manifest_path=args.fusion_manifest,
                output_dir=args.output_dir,
                frame_stride=max(1, int(args.frame_stride)),
                max_frames=args.max_frames,
                frame_indices=tuple(args.frame_indices) if args.frame_indices is not None else None,
                protocol_profile=str(args.protocol_profile),
                max_points_per_frame=args.max_points_per_frame,
                foreground_only=not bool(args.include_background),
                random_seed=int(args.random_seed),
            )
        )
        print(json.dumps({"reart_sequence_dir": str(output_dir.resolve())}, indent=2))
        return 0

    if args.command == "remote-reart-run":
        from .perception.reart_adapter import RemoteReArtClient, RemoteReArtConfig

        output_dir = RemoteReArtClient(
            server_url=args.server_url,
            api_token=args.api_token,
            timeout_s=min(float(args.timeout_s), 120.0),
        ).run(
            RemoteReArtConfig(
                server_url=args.server_url,
                sequence_dir=args.sequence_dir,
                output_dir=args.output_dir,
                sequence_name=args.sequence_name,
                cano_idx=int(args.cano_idx),
                num_points=max(1, int(args.num_points)),
                num_parts=max(1, int(args.num_parts)),
                stage=args.stage,
                base_n_iter=max(1, int(args.base_n_iter)),
                kinematic_n_iter=max(1, int(args.kinematic_n_iter)),
                assign_iter=max(0, int(args.assign_iter)),
                snapshot_gap=max(1, int(args.snapshot_gap)),
                use_assign_loss=bool(args.use_assign_loss),
                use_flow_loss=bool(args.use_flow_loss),
                use_nproc=bool(args.use_nproc),
                timeout_s=float(args.timeout_s),
                poll_interval_s=float(args.poll_interval_s),
                api_token=args.api_token,
                download_result=not bool(args.remote_skip_download),
            )
        )
        print(json.dumps({"remote_reart_output_dir": str(output_dir.resolve())}, indent=2))
        return 0

    if args.command == "compare-articulation-backends":
        from .perception.particulate_adapter import (
            ArticulationBackendComparisonConfig,
            ArticulationBackendComparator,
        )

        output_path = ArticulationBackendComparator().compare(
            ArticulationBackendComparisonConfig(
                tracking_joint_inference_path=args.tracking_joint_inference,
                particulate_result_path=args.particulate_result,
                output_json=args.output_json,
            )
        )
        print(json.dumps({"articulation_backend_comparison": str(output_path.resolve())}, indent=2))
        return 0

    if args.command == "evaluate-kinematic-model":
        from .kinematics.evaluation import KinematicModelEvaluationConfig, KinematicModelEvaluator

        output_path = KinematicModelEvaluator().evaluate(
            KinematicModelEvaluationConfig(
                joint_inference_path=args.joint_inference,
                part_pose_path=args.part_poses,
                output_json=args.output_json,
            )
        )
        print(json.dumps({"kinematic_evaluation": str(output_path.resolve())}, indent=2))
        return 0

    if args.command == "evaluate-object-mask-kinematics":
        from .kinematics.evaluation import ObjectMaskKinematicEvaluationConfig, ObjectMaskKinematicEvaluator

        output_path = ObjectMaskKinematicEvaluator().evaluate(
            ObjectMaskKinematicEvaluationConfig(
                joint_inference_path=args.joint_inference,
                part_pose_path=args.part_poses,
                output_json=args.output_json,
                output_csv=args.output_csv,
                matching_metric=args.matching_metric,
            )
        )
        print(json.dumps({"object_mask_kinematic_evaluation": str(output_path.resolve())}, indent=2))
        return 0

    if args.command == "evaluate-local-split-candidates":
        from .kinematics.local_split_candidate_evaluation import (
            LocalSplitCandidateEvaluationConfig,
            LocalSplitCandidateEvaluator,
        )

        output_path = LocalSplitCandidateEvaluator().evaluate(
            LocalSplitCandidateEvaluationConfig(
                local_split_summary=args.local_split_summary,
                output_dir=args.output_dir,
                output_json=args.output_json,
                output_csv=args.output_csv,
                min_tracks_per_part=max(3, int(args.min_tracks_per_part)),
                mujoco_prior=args.mujoco_prior,
                robust_track_model_trim_ratio=max(0.0, min(0.8, float(args.robust_track_model_trim_ratio))),
                matching_metric=args.matching_metric,
            )
        )
        print(json.dumps({"local_split_candidate_evaluation": str(output_path.resolve())}, indent=2))
        return 0

    if args.command == "evaluate-feedforward-articulation":
        from .kinematics.feedforward_evaluation import (
            FeedforwardArticulationEvaluationConfig,
            FeedforwardArticulationEvaluator,
        )

        output_path = FeedforwardArticulationEvaluator().evaluate(
            FeedforwardArticulationEvaluationConfig(
                feedforward_root=args.feedforward_root,
                reference_evaluation=args.reference_evaluation,
                output_json=args.output_json,
                yaw_degrees=tuple(int(value) for value in args.yaw_degrees),
                angle_score_weight=float(args.angle_score_weight),
                position_score_weight=float(args.position_score_weight),
            )
        )
        print(json.dumps({"feedforward_articulation_evaluation": str(output_path.resolve())}, indent=2))
        return 0

    if args.command == "render-mujoco-masks":
        from .sim.mujoco_recorder import MuJoCoMaskRenderConfig, MuJoCoMaskRenderer

        episode_path = MuJoCoMaskRenderer(
            MuJoCoMaskRenderConfig(
                episode_path=args.episode,
                model_path=args.model,
                mask_format=args.mask_format,
                part_segmentation_masks=bool(args.part_segmentation_masks),
                output_episode_path=args.output_episode,
            )
        ).render()
        print(json.dumps({"episode_path": str(episode_path.resolve())}, indent=2))
        return 0

    if args.command == "import-rbo-recording":
        from .perception.rbo_adapter import RBORecordingImportConfig, RBORecordingImporter

        if args.frame_stride < 1:
            parser.error("--frame-stride must be a positive integer")
        if args.start_frame < 0:
            parser.error("--start-frame must be >= 0")
        if args.max_frames is not None and args.max_frames < 1:
            parser.error("--max-frames must be a positive integer when provided")
        if args.target_fps is not None and args.target_fps <= 0.0:
            parser.error("--target-fps must be positive when provided")
        if args.max_sync_delta_s < 0:
            parser.error("--max-sync-delta-s must be >= 0")
        episode_path = RBORecordingImporter().import_recording(
            RBORecordingImportConfig(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                object_id=args.object_id,
                category=args.category,
                frame_stride=int(args.frame_stride),
                start_frame=int(args.start_frame),
                max_frames=args.max_frames,
                target_fps=args.target_fps,
                max_sync_delta_s=float(args.max_sync_delta_s),
                mask_mode=str(args.mask_mode),
                mask_depth_percentile=float(args.mask_depth_percentile),
                mask_depth_margin_m=float(args.mask_depth_margin_m),
                min_depth_m=float(args.min_depth_m),
                max_depth_m=float(args.max_depth_m),
                force=bool(args.force),
            )
        )
        print(json.dumps({"episode_path": str(episode_path.resolve())}, indent=2))
        return 0

    if args.command == "import-rbo-recordings-batch":
        from .perception.rbo_adapter import RBORecordingBatchConfig, RBORecordingBatchImporter

        if args.jobs < 1:
            parser.error("--jobs must be a positive integer")
        if args.frame_stride < 1:
            parser.error("--frame-stride must be a positive integer")
        if args.start_frame < 0:
            parser.error("--start-frame must be >= 0")
        if args.max_frames is not None and args.max_frames < 1:
            parser.error("--max-frames must be a positive integer when provided")
        if args.target_fps is not None and args.target_fps <= 0.0:
            parser.error("--target-fps must be positive when provided")
        result = RBORecordingBatchImporter(
            RBORecordingBatchConfig(
                manifest_path=args.manifest,
                output_root=args.output_root,
                jobs=int(args.jobs),
                frame_stride=int(args.frame_stride),
                start_frame=int(args.start_frame),
                max_frames=args.max_frames,
                target_fps=args.target_fps,
                max_sync_delta_s=float(args.max_sync_delta_s),
                mask_mode=str(args.mask_mode),
                mask_depth_percentile=float(args.mask_depth_percentile),
                mask_depth_margin_m=float(args.mask_depth_margin_m),
                min_depth_m=float(args.min_depth_m),
                max_depth_m=float(args.max_depth_m),
                force=bool(args.force),
            )
        ).run()
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "compact-mujoco-recording":
        from .sim.mujoco_recorder import MuJoCoEpisodeCompactConfig, MuJoCoEpisodeCompactor

        result = MuJoCoEpisodeCompactor(
            MuJoCoEpisodeCompactConfig(
                episode_path=args.episode,
                remove_concat_dir=not bool(args.keep_concat_dir),
                remove_concat_video=bool(args.remove_concat_video),
                dry_run=bool(args.dry_run),
            )
        ).compact()
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "repack-mujoco-recording":
        from .sim.mujoco_recorder import MuJoCoEpisodeRepackConfig, MuJoCoEpisodeRepacker

        result = MuJoCoEpisodeRepacker(
            MuJoCoEpisodeRepackConfig(
                episode_path=args.episode,
                keep_originals=bool(args.keep_originals),
                dry_run=bool(args.dry_run),
            )
        ).repack()
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "record-mujoco":
        from .sim.mujoco_recorder import MuJoCoEpisodeRecorder, MuJoCoRecordConfig

        frame_count = int(args.frames)
        frame_dt = float(args.frame_dt)
        if args.duration_s is not None or args.fps is not None:
            if args.duration_s is None or args.fps is None:
                parser.error("--duration-s and --fps must be provided together")
            if args.duration_s <= 0.0:
                parser.error("--duration-s must be positive")
            if args.fps <= 0.0:
                parser.error("--fps must be positive")
            frame_dt = 1.0 / float(args.fps)
            frame_count = max(1, int(round(float(args.duration_s) * float(args.fps))))

        if args.kick_start_s < 0.0:
            parser.error("--kick-start-s must be >= 0")
        if args.kick_duration_s < 0.0:
            parser.error("--kick-duration-s must be >= 0")
        if args.kick_duration_s == 0.0 and args.kick_force != 0.0:
            parser.error("--kick-force requires --kick-duration-s > 0")
        if args.excitation_start_s < 0.0:
            parser.error("--excitation-start-s must be >= 0")
        if args.excitation_duration_s < 0.0:
            parser.error("--excitation-duration-s must be >= 0")
        if args.excitation_mode != "none" and args.excitation_duration_s <= 0.0:
            parser.error("--excitation-duration-s must be > 0 when --excitation-mode is not none")
        if args.excitation_period_s <= 0.0:
            parser.error("--excitation-period-s must be positive")
        if args.excitation_frequency_hz <= 0.0:
            parser.error("--excitation-frequency-hz must be positive")
        if args.staged_initial_qvel:
            if args.control_mode != "free":
                parser.error("--staged-initial-qvel requires --control-mode free")
            if args.joint_name is not None or args.joint_id is not None:
                parser.error("--staged-initial-qvel cannot be combined with --joint-name/--joint-id")
            if args.staged_secondary_start_s < 0.0:
                parser.error("--staged-secondary-start-s must be >= 0")
            if args.staged_secondary_count_min < 0:
                parser.error("--staged-secondary-count-min must be >= 0")
            if args.staged_secondary_count_max < args.staged_secondary_count_min:
                parser.error("--staged-secondary-count-max must be >= --staged-secondary-count-min")

        if args.all_joints and (args.joint_name is not None or args.joint_id is not None):
            parser.error("--all-joints cannot be combined with --joint-name/--joint-id")
        if args.control_mode in {"staggered", "paper_sequential", "paper_simultaneous"} and (
            args.joint_name is not None or args.joint_id is not None
        ):
            parser.error(
                "--control-mode staggered/paper_sequential/paper_simultaneous cannot be combined with "
                "--joint-name/--joint-id"
            )
        if args.control_mode in {"paper_sequential", "paper_simultaneous"} and not args.paper_simultaneous_joint:
            parser.error("paper_sequential/paper_simultaneous requires --paper-simultaneous-joint")
        if args.control_mode not in {"paper_sequential", "paper_simultaneous"} and args.paper_simultaneous_joint:
            parser.error("--paper-simultaneous-joint requires paper_sequential/paper_simultaneous")
        if args.joint_name is not None and args.joint_id is not None:
            parser.error("Provide only one of --joint-name or --joint-id")

        config = MuJoCoRecordConfig(
            model_path=args.model,
            output_dir=args.output_dir,
            object_instance_id=args.object_id,
            category=args.category,
            frame_count=frame_count,
            sim_dt=args.sim_dt,
            frame_dt=frame_dt,
            width=args.width,
            height=args.height,
            rgb_format=args.rgb_format,
            depth_format=args.depth_format,
            write_concat_assets=bool(args.write_concat_assets),
            camera_distance=args.camera_distance,
            auto_camera_fit=bool(args.auto_camera_fit),
            camera_fit_fill_ratio=float(args.camera_fit_fill_ratio),
            camera_elevation_deg=args.camera_elevation_deg,
            camera_azimuth_start_deg=args.camera_azimuth_start_deg,
            camera_azimuth_span_deg=args.camera_azimuth_span_deg,
            camera_fovy_deg=args.camera_fovy_deg,
            camera_mode=args.camera_mode,
            camera_triview_spacing_deg=args.camera_triview_spacing_deg,
            recording_protocol=args.recording_protocol,
            aim_static_scan_views=int(args.aim_static_scan_views),
            aim_end_scan_views=int(args.aim_end_scan_views),
            aim_static_scan_elevation_deg=float(args.aim_static_scan_elevation_deg),
            aim_interaction_camera_orbits=float(args.aim_interaction_camera_orbits),
            aim_interaction_elevation_amplitude_deg=float(
                args.aim_interaction_elevation_amplitude_deg
            ),
            aim_interaction_camera_trajectory=args.interaction_camera_trajectory,
            aim_interaction_motion_end_fraction=float(
                args.aim_interaction_motion_end_fraction
            ),
            aim_interaction_fixed_view_azimuths_deg=tuple(
                args.aim_interaction_fixed_view_azimuths_deg
            ),
            aim_interaction_fixed_view_elevations_deg=tuple(
                args.aim_interaction_fixed_view_elevations_deg
            ),
            lookat=tuple(args.lookat),
            perturbation_scale=args.perturbation_scale,
            control_kp=args.control_kp,
            control_kd=args.control_kd,
            staggered_max_acceleration=args.staggered_max_acceleration,
            seed=args.seed,
            control_mode=args.control_mode,
            paper_simultaneous_joints=tuple(
                (str(name), float(start_q), float(end_q))
                for name, start_q, end_q in args.paper_simultaneous_joint
            ),
            random_initial_qpos=bool(args.random_initial_qpos),
            auto_initial_qvel_from_limits=bool(args.auto_initial_qvel_from_limits),
            auto_initial_qvel_direction_mode=args.auto_initial_qvel_direction_mode,
            auto_initial_qvel_min_abs=float(args.auto_initial_qvel_min_abs),
            auto_initial_qvel_max_abs=float(args.auto_initial_qvel_max_abs),
            initial_joint_qvel=args.initial_qvel,
            staged_initial_qvel=bool(args.staged_initial_qvel),
            staged_primary_tokens=tuple(str(token) for token in args.staged_primary_tokens),
            staged_secondary_tokens=tuple(str(token) for token in args.staged_secondary_tokens),
            staged_secondary_count_min=int(args.staged_secondary_count_min),
            staged_secondary_count_max=int(args.staged_secondary_count_max),
            staged_secondary_start_s=float(args.staged_secondary_start_s),
            kick_force=args.kick_force,
            kick_start_s=args.kick_start_s,
            kick_duration_s=args.kick_duration_s,
            excitation_mode=args.excitation_mode,
            excitation_force=float(args.excitation_force),
            excitation_start_s=float(args.excitation_start_s),
            excitation_duration_s=float(args.excitation_duration_s),
            excitation_period_s=float(args.excitation_period_s),
            excitation_frequency_hz=float(args.excitation_frequency_hz),
            write_dynamics_log=not bool(args.no_dynamics_log),
            make_video=bool(args.video),
            video_fps=args.video_fps,
            segmentation_masks=bool(args.segmentation_masks),
            part_segmentation_masks=bool(args.part_segmentation_masks),
            mask_format=args.mask_format,
            disable_target_mesh_collision=bool(args.disable_target_mesh_collision),
            disable_target_collision=bool(args.disable_target_collision),
            disable_gravity=bool(args.disable_gravity),
            hide_clear_meshes=bool(args.hide_clear_meshes),
            joint_name=args.joint_name,
            joint_id=args.joint_id,
            all_joints=bool(
                args.all_joints
                or args.control_mode in {"staggered", "paper_sequential", "paper_simultaneous"}
            ),
        )
        episode_path = MuJoCoEpisodeRecorder(config).record()
        print(json.dumps({"episode_path": str(episode_path.resolve())}, indent=2))
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2
