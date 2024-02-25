# run the same model on different objects
scene_name=scene0008_00
gpu_id=5
for obj_id in 8
do
    for exp_name in inpainted
    do
        python ./exp_runner.py --mode train --conf ./confs/paper/scannet-${exp_name}-default-stage1.conf --gpu ${gpu_id} --scene_name ${scene_name}_scannet_obj_${obj_id}_${exp_name}

        python ./exp_runner.py --mode train --conf ./confs/paper/scannet-${exp_name}-default-stage2.conf --gpu ${gpu_id} --scene_name ${scene_name}_scannet_obj_${obj_id}_${exp_name} --is_continue

        python ./exp_runner.py --mode validate_mesh --conf ./confs/paper/scannet-${exp_name}-default-stage2.conf --gpu ${gpu_id} --scene_name ${scene_name}_scannet_obj_${obj_id}_${exp_name} --is_continue

    done
done
