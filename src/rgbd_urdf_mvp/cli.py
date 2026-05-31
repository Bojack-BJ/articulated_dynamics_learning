from __future__ import annotations

import json
import sys

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
                fuse_pixel_stride=max(1, int(args.fuse_pixel_stride)),
                fuse_voxel_size_m=float(args.fuse_voxel_size_m),
                min_tracks_per_part=max(3, int(args.min_tracks_per_part)),
                mujoco_prior_mode=str(args.mujoco_prior),
                generate_viewer=not bool(args.no_generate_viewer),
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
                visibility_threshold=float(args.visibility_threshold),
                require_part_mask_consistency=not bool(args.no_part_mask_consistency),
                allow_backward_tracking=not bool(args.no_backward_tracking),
                show_progress=not bool(args.no_progress),
            )
        ).track()
        print(json.dumps({"part_tracks_artifact": str(output_json.resolve())}, indent=2))
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

    if args.command == "infer-joints":
        from .kinematics.joint_inference import JointInferenceConfig, JointInferencer

        output_json = JointInferencer(
            JointInferenceConfig(
                input_path=args.input,
                output_json=args.output_json,
                rotation_threshold_rad=float(args.rotation_threshold_rad),
                translation_threshold_m=float(args.translation_threshold_m),
                mujoco_prior=args.mujoco_prior,
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
            )
        )
        print(json.dumps({"remote_articulation_output_dir": str(output_dir.resolve())}, indent=2))
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

        if args.all_joints and (args.joint_name is not None or args.joint_id is not None):
            parser.error("--all-joints cannot be combined with --joint-name/--joint-id")
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
            camera_elevation_deg=args.camera_elevation_deg,
            camera_azimuth_start_deg=args.camera_azimuth_start_deg,
            camera_azimuth_span_deg=args.camera_azimuth_span_deg,
            camera_fovy_deg=args.camera_fovy_deg,
            camera_mode=args.camera_mode,
            camera_triview_spacing_deg=args.camera_triview_spacing_deg,
            lookat=tuple(args.lookat),
            perturbation_scale=args.perturbation_scale,
            control_kp=args.control_kp,
            control_kd=args.control_kd,
            seed=args.seed,
            control_mode=args.control_mode,
            random_initial_qpos=bool(args.random_initial_qpos),
            auto_initial_qvel_from_limits=bool(args.auto_initial_qvel_from_limits),
            auto_initial_qvel_direction_mode=args.auto_initial_qvel_direction_mode,
            auto_initial_qvel_min_abs=float(args.auto_initial_qvel_min_abs),
            auto_initial_qvel_max_abs=float(args.auto_initial_qvel_max_abs),
            initial_joint_qvel=args.initial_qvel,
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
            hide_clear_meshes=bool(args.hide_clear_meshes),
            joint_name=args.joint_name,
            joint_id=args.joint_id,
            all_joints=bool(args.all_joints),
        )
        episode_path = MuJoCoEpisodeRecorder(config).record()
        print(json.dumps({"episode_path": str(episode_path.resolve())}, indent=2))
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2
