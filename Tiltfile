docker_build('streamposeml-image', './stream_pose_ml', build_args={'flask_debug': 'True'},
    live_update=[
        sync('./stream_pose_ml', '/app'),
        run('cd /app && pip install -r requirements.txt',
            trigger='./stream_pose_ml/requirements.txt'),

        run('touch ./api/app.py'),

])

k8s_yaml('kubernetes.yml')
k8s_resource('streamposeml', port_forwards=5001)