import unittest
import shutil

from stream_pose_ml.services.video_data_service import VideoDataService

from stream_pose_ml.services.annotation_transformer_service import (
    AnnotationTransformerService,
)


class TestAnnotationTransformerService(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.video_input_path = "./stream_pose_ml/test_videos"
        self.output_data_path = "./stream_pose_ml/tmp/data/keypoints"
        self.input_filename = "front.mp4"
        self.annotation_data = {
            "id": "63e10d737329c2fe92c8ae0a",
            "datasetId": "63bef4c53775a03d44271475",
            "url": "https://gate.dataloop.ai/api/v1/items/63e10d737329c2fe92c8ae0a",
            "dataset": "https://gate.dataloop.ai/api/v1/datasets/63bef4c53775a03d44271475",
            "createdAt": "2023-02-06T14:23:47.000Z",
            "dir": "/IKF/Aug 27 2:30pm/Front",
            "filename": "/IKF/Aug 27 2:30pm/Front/IKF_8.27_230pm_BW_Front5_P9.webm",
            "type": "file",
            "hidden": False,
            "metadata": {
                "system": {
                    "originalname": "IKF_8.27_230pm_BW_Front5_P9.webm",
                    "size": 978603,
                    "encoding": "7bit",
                    "taskStatusLog": [],
                    "mimetype": "video/webm",
                    "refs": [],
                    "isBinary": True,
                    "duration": 9.583,
                    "ffmpeg": {
                        "avg_frame_rate": "30000/1001",
                        "chroma_location": "left",
                        "closed_captions": 0,
                        "codec_long_name": "Google VP9",
                        "codec_name": "vp9",
                        "codec_tag": "0x0000",
                        "codec_tag_string": "[0][0][0][0]",
                        "codec_type": "video",
                        "coded_height": 1080,
                        "coded_width": 1920,
                        "color_primaries": "bt709",
                        "color_range": "tv",
                        "color_space": "bt709",
                        "color_transfer": "bt709",
                        "display_aspect_ratio": "16:9",
                        "disposition": {
                            "attached_pic": 0,
                            "clean_effects": 0,
                            "comment": 0,
                            "default": 1,
                            "dub": 0,
                            "forced": 0,
                            "hearing_impaired": 0,
                            "karaoke": 0,
                            "lyrics": 0,
                            "original": 0,
                            "timed_thumbnails": 0,
                            "visual_impaired": 0,
                        },
                        "field_order": "progressive",
                        "has_b_frames": 0,
                        "height": 1080,
                        "index": 0,
                        "level": -99,
                        "nb_read_frames": "287",
                        "nb_read_packets": "287",
                        "pix_fmt": "yuv420p",
                        "profile": "Profile 0",
                        "r_frame_rate": "30000/1001",
                        "refs": 1,
                        "sample_aspect_ratio": "1:1",
                        "start_pts": 7,
                        "start_time": "0.007000",
                        "tags": {
                            "DURATION": "00:00:09.583000000",
                            "ENCODER": "Lavc59.37.100 libvpx-vp9",
                            "HANDLER_NAME": "Core Media Video",
                            "VENDOR_ID": "[0][0][0][0]",
                        },
                        "time_base": "1/1000",
                        "width": 1920,
                    },
                    "fps": 29.97002997002997,
                    "height": 1080,
                    "nb_streams": 2,
                    "startTime": 0.007,
                    "width": 1920,
                    "thumbnailId": "63e10d7aeb77175585967f38",
                },
                "fps": 29.97002997002997,
                "startTime": 0.007,
            },
            "name": "IKF_8.27_230pm_BW_Front5_P9.webm",
            "creator": "trajkovamilka@gmail.com",
            "stream": "https://gate.dataloop.ai/api/v1/items/63e10d737329c2fe92c8ae0a/stream",
            "thumbnail": "https://gate.dataloop.ai/api/v1/items/63e10d737329c2fe92c8ae0a/thumbnail",
            "annotations": [
                {
                    "id": "63fe90715ff162c693fa0f3c",
                    "datasetId": "63bef4c53775a03d44271475",
                    "itemId": "63e10d737329c2fe92c8ae0a",
                    "url": "https://gate.dataloop.ai/api/v1/annotations/63fe90715ff162c693fa0f3c",
                    "item": "https://gate.dataloop.ai/api/v1/items/63e10d737329c2fe92c8ae0a",
                    "dataset": "https://gate.dataloop.ai/api/v1/datasets/63bef4c53775a03d44271475",
                    "type": "class",
                    "label": "Left Step",
                    "attributes": [],
                    "metadata": {
                        "system": {
                            "startTime": 5.472133333333334,
                            "endTime": 6.940266666666667,
                            "frame": 164,
                            "endFrame": 208,
                            "snapshots_": [],
                            "clientId": "594e78ed-8ea4-42c5-96cd-6757fa730a16",
                            "automated": False,
                            "objectId": "3",
                            "isOpen": False,
                            "isOnlyLocal": False,
                            "attributes": {},
                            "system": False,
                            "itemLinks": [],
                            "openAnnotationVersion": "1.56.0-prod.31",
                            "recipeId": "63bef4c5223e5c2a0a9e4227",
                        },
                        "user": {},
                    },
                    "creator": "trajkovamilka@gmail.com",
                    "createdAt": "2023-02-28T23:38:25.259Z",
                    "updatedBy": "trajkovamilka@gmail.com",
                    "updatedAt": "2023-02-28T23:38:25.259Z",
                    "hash": "126958046adf48aeba68b6a125f7d02812d58dce",
                    "source": "ui",
                },
                {
                    "id": "63fe90715ff1623838fa0f3a",
                    "datasetId": "63bef4c53775a03d44271475",
                    "itemId": "63e10d737329c2fe92c8ae0a",
                    "url": "https://gate.dataloop.ai/api/v1/annotations/63fe90715ff1623838fa0f3a",
                    "item": "https://gate.dataloop.ai/api/v1/items/63e10d737329c2fe92c8ae0a",
                    "dataset": "https://gate.dataloop.ai/api/v1/datasets/63bef4c53775a03d44271475",
                    "type": "class",
                    "label": "Right Step",
                    "attributes": [],
                    "metadata": {
                        "system": {
                            "startTime": 0.5005000000000001,
                            "endTime": 1.9352666666666667,
                            "frame": 15,
                            "endFrame": 58,
                            "snapshots_": [],
                            "clientId": "5220cc7a-dccb-4e28-8318-5823a2cf8596",
                            "automated": False,
                            "objectId": "1",
                            "isOpen": False,
                            "isOnlyLocal": True,
                            "attributes": {},
                            "system": True,
                            "itemLinks": [],
                            "openAnnotationVersion": "1.56.0-prod.31",
                            "recipeId": "63bef4c5223e5c2a0a9e4227",
                        },
                        "user": {},
                    },
                    "creator": "trajkovamilka@gmail.com",
                    "createdAt": "2023-02-28T23:38:25.255Z",
                    "updatedBy": "trajkovamilka@gmail.com",
                    "updatedAt": "2023-02-28T23:38:25.255Z",
                    "hash": "67378dc20d711351ce94c279c20bd576b4ef1405",
                    "source": "ui",
                },
                {
                    "id": "63fe90715ff1626fe6fa0f3b",
                    "datasetId": "63bef4c53775a03d44271475",
                    "itemId": "63e10d737329c2fe92c8ae0a",
                    "url": "https://gate.dataloop.ai/api/v1/annotations/63fe90715ff1626fe6fa0f3b",
                    "item": "https://gate.dataloop.ai/api/v1/items/63e10d737329c2fe92c8ae0a",
                    "dataset": "https://gate.dataloop.ai/api/v1/datasets/63bef4c53775a03d44271475",
                    "type": "class",
                    "label": "Failure Weight Transfer",
                    "attributes": [],
                    "metadata": {
                        "system": {
                            "startTime": 0.5005000000000001,
                            "endTime": 9.567891666666666,
                            "frame": 15,
                            "endFrame": 286,
                            "snapshots_": [],
                            "clientId": "940ae3f8-4cfd-40a7-be0c-c592aaafcf25",
                            "automated": True,
                            "objectId": "2",
                            "isOpen": True,
                            "isOnlyLocal": True,
                            "attributes": {},
                            "system": True,
                            "itemLinks": [],
                            "openAnnotationVersion": "1.56.0-prod.31",
                            "recipeId": "63bef4c5223e5c2a0a9e4227",
                        },
                        "user": {},
                    },
                    "creator": "trajkovamilka@gmail.com",
                    "createdAt": "2023-02-28T23:38:25.257Z",
                    "updatedBy": "trajkovamilka@gmail.com",
                    "updatedAt": "2023-02-28T23:38:25.257Z",
                    "hash": "954fd42004669684bf673eb4fa4a8f68cc8e3008",
                    "source": "ui",
                },
            ],
            "annotationsCount": 3,
            "annotated": True,
        }

    @classmethod
    def tearDownClass(self) -> None:
        # cleanup
        try:
            shutil.rmtree(self.output_data_path)
        except:
            return super().tearDown(self)

        return super().tearDown(self)

    def test_segment_video_data_with_annotations(self):
        """
        GIVEN processed video data and corresponding data loop annotation data
        WHEN passed into AnnotationTransformer
        THEN a video annotations object is returned marrying the data annotations with corresponding frame data
        """
        vds = VideoDataService()
        video_data = vds.process_video(
            input_filename=self.input_filename,
            video_input_path=self.video_input_path,
            output_data_path=self.output_data_path,
            include_geometry=True,
        )

        transformer = AnnotationTransformerService()
        segmented_video_annotations = transformer.segment_video_data_with_annotations(
            annotation_data=self.annotation_data, video_data=video_data
        )
        # make sure we have the same number of annotations and video clips
        labels = [
            annotation["label"] for annotation in self.annotation_data["annotations"]
        ]
        annotation_count = 0
        for label in labels:
            annotation_count += len(segmented_video_annotations[label])
        
        self.assertEqual(3, annotation_count)
        # TODO fix this one to match segmented annotation structure
        # frame_length = [
        #     (
        #         annotation["metadata"]["system"]["endFrame"]
        #         - annotation["metadata"]["system"]["frame"]
        #     )
        #     for annotation in self.annotation_data["annotations"]
        # ]
        # for l, label in zip(frame_length, labels):
        #     self.assertEqual(l, len(segmented_video_annotations[label]) - 1)
