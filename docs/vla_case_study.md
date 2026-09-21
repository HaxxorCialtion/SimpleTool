# SimpleTool-VLA grasp case: what the video does and does not show

The accompanying video shows a successful MuJoCo Panda red-cube grasp closed loop. The run used an explicit task prompt and a calibrated RGB-D localization tool.

## What was provided to the model

The prompt supplied the tabletop height, approximate cube dimensions, approach order, grasp height, gripper settings, the `locate_red_cube` tool, and a final lift height. The tool description also exposed a known cube-height assumption and returned a cube-center estimate.

The observed action sequence was:

`locate_red_cube → move above cube → descend to z≈0.235 → close gripper → lift → finish`

## Valid claim

> SimpleTool-VLA completed a tool-mediated Panda grasping loop under explicit task guidance and calibrated RGB-D localization assistance.

The video demonstrates image/state-to-tool execution, multi-step action formatting, localization, gripper closure, exploratory lift, and completion detection.

## Claims we do not make

This run is not evidence that the model independently discovered the grasp order, inferred the grasp height, chose when to close the gripper, or planned the trajectory without assistance. It is not a zero-shot autonomous-planning result and it is not a pure-vision-only result.

The video is included as an auditable systems case study. The prompt-assisted setup is intentional: it makes the tool protocol and executor loop visible while keeping the autonomy boundary explicit.
