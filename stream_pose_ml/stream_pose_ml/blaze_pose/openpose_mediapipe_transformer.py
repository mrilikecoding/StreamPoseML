import stream_pose_ml.blaze_pose.blaze_pose_frame as bpf


class OpenPoseMediapipeTransformer:
    """Compute common angle and distance measures, typically used when doing CV with OpenPose .

    This class is responsible for encapsulating methods to make translation
    between Openpose angle and distance paradigms to equivalent Mediapipe
    representations. Openpose uses the Body 25 model. To represent the plumbline
    in Openpose, the vector between joints 25 and 26 is specified, the neck and mid_hip.
    Mediapipe uses BlazePose, so the plumbline can be represented by calculating the
    midpoint between the shoulders and the midpoint between the hip and representing
    as a vector.
    """

    @staticmethod
    def open_pose_distance_definition_map() -> dict:
        """A map of OpenPose distance definitions to vector/joint names used in this package.

        This method is responsible for translating Blaze pose joints into
        OpenPose domain joints / vectors and returning a map to vectors
        that can be used to generate distance calculations between a joint
        and vectors

        Returns:
            distance_definition_map: dict
                A map from named joints to vectors for use in
                distance calculation


        More info - here are the OpenPose distance measurements as they are named in OP docs
        as well as the tuple of joints / vectors as represented here.

        "distance_point_0__line_25_26": ("nose", "plumb_line"),
        "distance_point_1__line_25_26": ("neck", "plumb_line"),
        "distance_point_2__line_25_26": ("right_shoulder", "plumb_line"),
        "distance_point_3__line_25_26": ("right_elbow", "plumb_line"),
        "distance_point_4__line_25_26": ("right_wrist", "plumb_line"),
        "distance_point_5__line_25_26": ("left_shoulder", "plumb_line"),
        "distance_point_6__line_25_26": ("left_elbow", "plumb_line"),
        "distance_point_7__line_25_26": ("left_wrist", "plumb_line"),
        "distance_point_8__line_25_26": ("mid_hip", "plumb_line"),
        "distance_point_9__line_25_26": ("right_hip", "plumb_line"),
        "distance_point_10__line_25_26": ("right_knee", "plumb_line"),
        "distance_point_11__line_25_26": ("right_ankle", "plumb_line"),
        "distance_point_12__line_25_26": ("left_hip", "plumb_line"),
        "distance_point_13__line_25_26": ("left_knee", "plumb_line"),
        "distance_point_14__line_25_26": ("left_ankle", "plumb_line"),
        "distance_point_15__line_25_26": ("right_eye", "plumb_line"),
        "distance_point_16__line_25_26": ("left_eye", "plumb_line"),
        "distance_point_17__line_25_26": ("right_ear", "plumb_line"),
        "distance_point_18__line_25_26": ("left_ear", "plumb_line"),
        "distance_point_19__line_25_26": ("left_foot_index", "plumb_line"),
        # "distance_point_20__line_25_26": ("", "plumb_line"), # skipping as there's not a good analog
        "distance_point_21__line_25_26": ("left_heel", "plumb_line"),
        "distance_point_22__line_25_26": ("right_foot_index", "plumb_line"),
        # "distance_point_23__line_25_26": ("", "plumb_line"), #skipping as there's not a good analog
        "distance_point_24__line_25_26": ("right_heel", "plumb_line"),

        """
        return {
            "nose_to_plumb_line": ("nose", "plumb_line"),
            "neck_to_plumb_line": ("neck", "plumb_line"),
            "right_shoulder_to_plumb_line": ("right_shoulder", "plumb_line"),
            "right_elbow_to_plumb_line": ("right_elbow", "plumb_line"),
            "right_wrist_to_plumb_line": ("right_wrist", "plumb_line"),
            "left_shoulder_to_plumb_line": ("left_shoulder", "plumb_line"),
            "left_elbow_to_plumb_line": ("left_elbow", "plumb_line"),
            "left_wrist_to_plumb_line": ("left_wrist", "plumb_line"),
            "mid_hip_to_plumb_line": ("mid_hip", "plumb_line"),
            "right_hip_to_plumb_line": ("right_hip", "plumb_line"),
            "right_knee_to_plumb_line": ("right_knee", "plumb_line"),
            "right_ankle_to_plumb_line": ("right_ankle", "plumb_line"),
            "left_hip_to_plumb_line": ("left_hip", "plumb_line"),
            "left_knee_to_plumb_line": ("left_knee", "plumb_line"),
            "left_ankle_to_plumb_line": ("left_ankle", "plumb_line"),
            "right_eye_to_plumb_line": ("right_eye", "plumb_line"),
            "left_eye_to_plumb_line": ("left_eye", "plumb_line"),
            "right_ear_to_plumb_line": ("right_ear", "plumb_line"),
            "left_ear_to_plumb_line": ("left_ear", "plumb_line"),
            "left_foot_index_to_plumb_line": ("left_foot_index", "plumb_line"),
            "left_heel_to_plumb_line": ("left_heel", "plumb_line"),
            "right_foot_index_to_plumb_line": ("right_foot_index", "plumb_line"),
            "right_heel_to_plumb_line": ("right_heel", "plumb_line"),
        }

    @staticmethod
    def open_pose_angle_definition_map() -> dict:
        """A map of OpenPose angle definitions to vector/joint names used in this package.

        This method is responsible for translating Mediapipe Blaze pose joints into
        OpenPose domain joints / vectors and returning a map to vectors
        that can be used to generate angle calculations between vectors

        Returns:
            angle_definition_map: dict
                A map from named angles to created vectors for use in
                angle generation

        More info:

        More info - here are the OpenPose angle measurements as they are named in OP docs
        as well as the tuple of joints / vectors as represented here.

        "line_0_1__line_25_26": ("nose_neck", "plumb_line"),
        "line_1_8__line_25_26": ("neck_mid_hip", "plumb_line"),
        "line_1_2__line_25_26": ("neck_right_shoulder", "plumb_line"),
        "line_1_5__line_25_26": ("neck_left_shoulder", "plumb_line"),
        "line_2_3__line_25_26": ("right_shoulder_right_elbow", "plumb_line"),
        "line_3_4__line_25_26": ("right_elbow_right_wrist", "plumb_line"),
        "line_5_6__line_25_26": ("left_shoulder_left_elbow", "plumb_line"),
        "line_6_7__line_25_26": ("right_elbow_right_wrist", "plumb_line"),
        "line_8_9__line_25_26": ("mid_hip_right_hip", "plumb_line"),
        "line_9_10__line_25_26": ("right_hip_right_knee", "plumb_line"),
        "line_10_11__line_25_26": ("right_knee_right_ankle", "plumb_line"),
        "line_8_12__line_25_26": ("mid_hip_left_hip", "plumb_line"),
        "line_12_13__line_25_26": ("left_hip_left_knee", "plumb_line"),
        "line_13_14__line_25_26": ("left_knee_left_ankle", "plumb_line"),
        "line_0_15__line_25_26": ("nose_right_eye", "plumb_line"),
        "line_15_17__line_25_26": ("right_eye_right_ear", "plumb_line"),
        "line_0_16__line_25_26": ("nose_left_eye", "plumb_line"),
        "line_16_18__line_25_26": ("left_eye_left_ear", "plumb_line"),
        "line_14_19__line_25_26": ("left_ankle_left_foot_index", "plumb_line"),
        "line_14_21__line_25_26": ("left_ankle_left_heel", "plumb_line"),
        "line_11_22__line_25_26": ("right_ankle_right_foot_index", "plumb_line"),
        "line_11_24__line_25_26": ("right_ankle_right_heel", "plumb_line"),
        "line_0_1__line_1_8": ("right_ankle_right_heel", "plumb_line"),
        "line_0_1__line_1_2": ("nose_neck", "neck_right_shoulder"),
        "line_1_8__line_8_12": ("neck_mid_hip", "mid_hip_left_hip"),
        "line_1_2__line_2_3": ("neck_right_shoulder", "right_shoulder_right_elbow"),
        "line_1_5__line_5_6": ("neck_left_shoulder", "left_shoulder_left_elbow"),
        "line_2_3__line_3_4": (
            "right_shoulder_right_elbow",
            "plumb_line",
        ),
        "line_5_6__line_6_7": (
            "left_shoulder_left_elbow",
            "right_elbow_right_wrist",
        ),
        "line_8_9__line_9_10": ("mid_hip_right_hip", "right_hip_right_knee"),
        "line_10_11__line_11_24": (
            "right_knee_right_ankle",
            "right_ankle_right_heel",
        ),
        "line_8_12__line_12_13": ("mid_hip_left_hip", "left_hip_left_knee"),
        "line_12_13__line_13_14": ("left_hip_left_knee", "left_knee_left_ankle"),
        "line_13_14__line_14_19": (
            "left_knee_left_ankle",
            "left_ankle_left_foot_index",
        ),
        "line_0_15__line_15_17": ("nose_right_eye", "right_eye_right_ear"),
        "line_0_16__line_16_18": ("nose_left_eye", "left_eye_left_ear"),
        "line_0_16__line_16_18": ("nose_left_eye", "left_eye_left_ear"),
        "line_14_21__line_11_22": (
            "left_ankle_left_heel",
            "right_ankle_right_foot_index",
        ),
        """
        # create a translation map from openpose angle spec to vector->vector angle
        return {
            "nose_neck_to_plumb_line": ("nose_neck", "plumb_line"),
            "neck_mid_hip_to_plumb_line": ("neck_mid_hip", "plumb_line"),
            "neck_right_shoulder_to_plumb_line": ("neck_right_shoulder", "plumb_line"),
            "neck_left_shoulder_to_plumb_line": ("neck_left_shoulder", "plumb_line"),
            "right_shoulder_right_elbow_to_plumb_line": (
                "right_shoulder_right_elbow",
                "plumb_line",
            ),
            "right_elbow_right_wrist_to_plumb_line": (
                "right_elbow_right_wrist",
                "plumb_line",
            ),
            "left_shoulder_left_elbow_to_plumb_line": (
                "left_shoulder_left_elbow",
                "plumb_line",
            ),
            "right_elbow_right_wrist_to_plumb_line": (
                "right_elbow_right_wrist",
                "plumb_line",
            ),
            "mid_hip_right_hip_to_plumb_line": ("mid_hip_right_hip", "plumb_line"),
            "right_hip_right_knee_to_plumb_line": (
                "right_hip_right_knee",
                "plumb_line",
            ),
            "right_knee_right_ankle_to_plumb_line": (
                "right_knee_right_ankle",
                "plumb_line",
            ),
            "mid_hip_left_hip_to_plumb_line": ("mid_hip_left_hip", "plumb_line"),
            "left_hip_left_knee_to_plumb_line": ("left_hip_left_knee", "plumb_line"),
            "left_knee_left_ankle_to_plumb_line": (
                "left_knee_left_ankle",
                "plumb_line",
            ),
            "nose_right_eye_to_plumb_line": ("nose_right_eye", "plumb_line"),
            "right_eye_right_ear_to_plumb_line": ("right_eye_right_ear", "plumb_line"),
            "nose_left_eye_to_plumb_line": ("nose_left_eye", "plumb_line"),
            "left_eye_left_ear_to_plumb_line": ("left_eye_left_ear", "plumb_line"),
            "left_ankle_left_foot_index_to_plumb_line": (
                "left_ankle_left_foot_index",
                "plumb_line",
            ),
            "left_ankle_left_heel_to_plumb_line": (
                "left_ankle_left_heel",
                "plumb_line",
            ),
            "right_ankle_right_foot_index_to_plumb_line": (
                "right_ankle_right_foot_index",
                "plumb_line",
            ),
            "right_ankle_right_heel_to_plumb_line": (
                "right_ankle_right_heel",
                "plumb_line",
            ),
            "right_ankle_right_heel_to_neck_mid_hip": (
                "right_ankle_right_heel",
                "neck_mid_hip",
            ),
            "nose_neck_to_neck_right_shoulder": ("nose_neck", "neck_right_shoulder"),
            "neck_mid_hip_to_mid_hip_left_hip": ("neck_mid_hip", "mid_hip_left_hip"),
            "neck_right_shoulder_to_right_shoulder_right_elbow": (
                "neck_right_shoulder",
                "right_shoulder_right_elbow",
            ),
            "neck_left_shoulder_to_left_shoulder_left_elbow": (
                "neck_left_shoulder",
                "left_shoulder_left_elbow",
            ),
            "right_shoulder_right_elbow_to_plumb_line": (
                "right_shoulder_right_elbow",
                "plumb_line",
            ),
            "left_shoulder_left_elbow_to_right_elbow_right_wrist": (
                "left_shoulder_left_elbow",
                "right_elbow_right_wrist",
            ),
            "mid_hip_right_hip_to_right_hip_right_knee": (
                "mid_hip_right_hip",
                "right_hip_right_knee",
            ),
            "right_knee_right_ankle_to_right_ankle_right_heel": (
                "right_knee_right_ankle",
                "right_ankle_right_heel",
            ),
            "mid_hip_left_hip_to_left_hip_left_knee": (
                "mid_hip_left_hip",
                "left_hip_left_knee",
            ),
            "left_hip_left_knee_to_left_knee_left_ankle": (
                "left_hip_left_knee",
                "left_knee_left_ankle",
            ),
            "left_knee_left_ankle_to_left_ankle_left_foot_index": (
                "left_knee_left_ankle",
                "left_ankle_left_foot_index",
            ),
            "nose_right_eye_to_right_eye_right_ear": (
                "nose_right_eye",
                "right_eye_right_ear",
            ),
            "nose_left_eye_to_left_eye_left_ear": (
                "nose_left_eye",
                "left_eye_left_ear",
            ),
            "left_ankle_left_heel_to_right_ankle_right_foot_index": (
                "left_ankle_left_heel",
                "right_ankle_right_foot_index",
            ),
        }

    @staticmethod
    def create_openpose_joints_and_vectors(
        frame: "bpf.BlazePoseFrame",
    ) -> bool:
        """Create new joints / vectors based on OpenPose angles/distance.

        This method is responsible for creating new joints that are
        part of the OpenPose 25 Body Key-points. The purpose of doing this
        is to use body angle measurements that are consistent with angles used
        with other projects that have used OpenPose for pose estimation.

        Here new joints are created by averaging coodinate information
        from Blaze points to create intermediate points. Then vectors are
        calculated to use as the basis for angle calculations.

        Args:
            frame: BlazePoseFrame - a BlazePoseFrame object to be updated

        Returns:
            success: bool
                Returns: True if joints and vectors are created successfully,
                False otherwise

        Raises:
            exception: BlazePoseFrameError
                Raises: BlazePoseFrameError if there is a problem

        """
        if not frame.has_joint_positions:
            return False

        try:
            frame.joints["neck"] = frame.get_average_joint(
                name="neck", joint_1="left_shoulder", joint_2="right_shoulder"
            )
            frame.joints["mid_hip"] = frame.get_average_joint(
                name="mid_hip", joint_1="left_hip", joint_2="right_hip"
            )

            # create necessary vectors translating from openpose domain
            # i.e. create plumb line vector as the angle basis - joints must
            # exist in self.joints

            # OpenPose 25_26
            frame.vectors["plumb_line"] = frame.get_vector(
                "plumb_line", "neck", "mid_hip"
            )
            # OpenPose 0_1
            frame.vectors["nose_neck"] = frame.get_vector("nose_neck", "nose", "neck")
            # OpenPose 1_8
            frame.vectors["neck_mid_hip"] = frame.get_vector(
                "neck_mid_hip", "neck", "mid_hip"
            )
            # OpenPose 1_2
            frame.vectors["neck_right_shoulder"] = frame.get_vector(
                "neck_right_shoulder", "neck", "right_shoulder"
            )
            # OpenPose 1_5
            frame.vectors["neck_left_shoulder"] = frame.get_vector(
                "neck_left_shoulder", "neck", "left_shoulder"
            )
            # OpenPose 2_3
            frame.vectors["right_shoulder_right_elbow"] = frame.get_vector(
                "right_shoulder_right_elbow", "right_shoulder", "right_elbow"
            )
            # OpenPose 3_4
            frame.vectors["right_elbow_right_wrist"] = frame.get_vector(
                "right_elbow_right_wrist", "right_elbow", "right_wrist"
            )
            # OpenPose 5_6
            frame.vectors["left_shoulder_left_elbow"] = frame.get_vector(
                "left_shoulder_left_elbow", "left_shoulder", "left_elbow"
            )
            # OpenPose 6_7
            frame.vectors["right_elbow_right_wrist"] = frame.get_vector(
                "left_elbow_left_wrist", "left_elbow", "left_wrist"
            )
            # OpenPose 8_9
            frame.vectors["mid_hip_right_hip"] = frame.get_vector(
                "mid_hip_right_hip", "mid_hip", "right_hip"
            )
            # OpenPose 9_10
            frame.vectors["right_hip_right_knee"] = frame.get_vector(
                "right_hip_right_knee", "right_hip", "right_knee"
            )
            # OpenPose 10_11
            frame.vectors["right_knee_right_ankle"] = frame.get_vector(
                "right_knee_right_ankle", "right_knee", "right_ankle"
            )
            # OpenPose 8_12
            frame.vectors["mid_hip_left_hip"] = frame.get_vector(
                "mid_hip_left_hip", "mid_hip", "left_hip"
            )
            # OpenPose 12_13
            frame.vectors["left_hip_left_knee"] = frame.get_vector(
                "left_hip_left_knee", "left_hip", "left_knee"
            )
            # OpenPose 13_14
            frame.vectors["left_knee_left_ankle"] = frame.get_vector(
                "left_knee_left_ankle", "left_knee", "left_ankle"
            )
            # OpenPose 0_15
            frame.vectors["nose_right_eye"] = frame.get_vector(
                "nose_right_eye", "nose", "right_eye"
            )
            # OpenPose 15_17
            frame.vectors["right_eye_right_ear"] = frame.get_vector(
                "right_eye_right_ear", "right_eye", "right_ear"
            )
            # OpenPose 0_16
            frame.vectors["nose_left_eye"] = frame.get_vector(
                "nose_left_eye", "nose", "left_eye"
            )
            # OpenPose 16_18
            frame.vectors["left_eye_left_ear"] = frame.get_vector(
                "left_eye_left_ear", "left_eye", "left_ear"
            )
            # OpenPose 14_19 -- note OpenPose is "Left Big Toe" - here using left foot index from Blaze
            frame.vectors["left_ankle_left_foot_index"] = frame.get_vector(
                "left_ankle_left_foot_index", "left_ankle", "left_foot_index"
            )
            # OpenPose 19_20 -- Skipping - not a good analog for this
            # OpenPose 14_21
            frame.vectors["left_ankle_left_heel"] = frame.get_vector(
                "left_ankle_left_heel", "left_ankle", "left_heel"
            )
            # OpenPose 11_22 -- note OpenPose is "Right Big Toe" - here using right foot index from Blaze
            frame.vectors["right_ankle_right_foot_index"] = frame.get_vector(
                "right_ankle_right_foot_index", "right_ankle", "right_foot_index"
            )
            # OpenPose 22_23 -- Skipping - not a good analog for this
            # OpenPose 11_24
            frame.vectors["right_ankle_right_heel"] = frame.get_vector(
                "right_ankle_right_heel", "right_ankle", "right_heel"
            )
        except:
            raise OpenPoseMediapipeTransformerError("Problem setting joints or vectors")

        return True


class OpenPoseMediapipeTransformerError(Exception):
    """
    Raised when there is an error in the OpenPoseMediapipeTranformer class
    """

    pass
