class DataloopAnnotationTransformerService:
    """
    This class is responsible for marrying video frame data and annotations
    """

    data: dict

    def __init__(self, dataloop_data) -> None:
        """
        Upon initialization this class sets a data object to the passed dataloop data dictionary

        Parameters
        ---------

                dataloop_data: dict
                        This is a dataloop dictionary (see schema at bottom of file)


        """
        self.data = dataloop_data

    def segment_video_data_with_annotations(self, video_data: dict) -> dict:
        segmented_video_data = {}
        for annotation in self.data["annotations"]:
            label = annotation["label"]
            segmented_video_data[label] = {}
            start_frame = annotation["metadata"]["system"]["frame"]
            end_frame = annotation["metadata"]["system"]["endFrame"]
            for frame_number in range(start_frame, end_frame + 1):
                segmented_video_data[label][frame_number] = video_data["frames"][
                    frame_number
                ]
        return segmented_video_data


"""
Schema
{
  "id": "63e10d737329c2fe92c8ae0a",
  "datasetId": "63bef4c53775a03d44271475",
  "url": "https://gate.dataloop.ai/api/v1/items/63e10d737329c2fe92c8ae0a",
  "dataset": "https://gate.dataloop.ai/api/v1/datasets/63bef4c53775a03d44271475",
  "createdAt": "2023-02-06T14:23:47.000Z",
  "dir": "/IKF/Aug 27 2:30pm/Front",
  "filename": "/IKF/Aug 27 2:30pm/Front/IKF_8.27_230pm_BW_Front5_P9.webm",
  "type": "file",
  "hidden": false,
  "metadata": {
    "system": {
      "originalname": "IKF_8.27_230pm_BW_Front5_P9.webm",
      "size": 978603,
      "encoding": "7bit",
      "taskStatusLog": [],
      "mimetype": "video/webm",
      "refs": [],
      "isBinary": true,
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
          "visual_impaired": 0
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
          "VENDOR_ID": "[0][0][0][0]"
        },
        "time_base": "1/1000",
        "width": 1920
      },
      "fps": 29.97002997002997,
      "height": 1080,
      "nb_streams": 2,
      "startTime": 0.007,
      "width": 1920,
      "thumbnailId": "63e10d7aeb77175585967f38"
    },
    "fps": 29.97002997002997,
    "startTime": 0.007
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
          "automated": false,
          "objectId": "3",
          "isOpen": false,
          "isOnlyLocal": false,
          "attributes": {},
          "system": false,
          "itemLinks": [],
          "openAnnotationVersion": "1.56.0-prod.31",
          "recipeId": "63bef4c5223e5c2a0a9e4227"
        },
        "user": {}
      },
      "creator": "trajkovamilka@gmail.com",
      "createdAt": "2023-02-28T23:38:25.259Z",
      "updatedBy": "trajkovamilka@gmail.com",
      "updatedAt": "2023-02-28T23:38:25.259Z",
      "hash": "126958046adf48aeba68b6a125f7d02812d58dce",
      "source": "ui"
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
          "automated": false,
          "objectId": "1",
          "isOpen": false,
          "isOnlyLocal": false,
          "attributes": {},
          "system": false,
          "itemLinks": [],
          "openAnnotationVersion": "1.56.0-prod.31",
          "recipeId": "63bef4c5223e5c2a0a9e4227"
        },
        "user": {}
      },
      "creator": "trajkovamilka@gmail.com",
      "createdAt": "2023-02-28T23:38:25.255Z",
      "updatedBy": "trajkovamilka@gmail.com",
      "updatedAt": "2023-02-28T23:38:25.255Z",
      "hash": "67378dc20d711351ce94c279c20bd576b4ef1405",
      "source": "ui"
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
          "automated": false,
          "objectId": "2",
          "isOpen": false,
          "isOnlyLocal": false,
          "attributes": {},
          "system": false,
          "itemLinks": [],
          "openAnnotationVersion": "1.56.0-prod.31",
          "recipeId": "63bef4c5223e5c2a0a9e4227"
        },
        "user": {}
      },
      "creator": "trajkovamilka@gmail.com",
      "createdAt": "2023-02-28T23:38:25.257Z",
      "updatedBy": "trajkovamilka@gmail.com",
      "updatedAt": "2023-02-28T23:38:25.257Z",
      "hash": "954fd42004669684bf673eb4fa4a8f68cc8e3008",
      "source": "ui"
    }
  ],
  "annotationsCount": 3,
  "annotated": true
}
"""
