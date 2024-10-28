# ---
1.数据准备
将省赛数据集解压放入data文件夹下，对于测试集A修改名称为test_A_joint.npy标签数据改为test_A_label.npy，对于测试集B修改名称为test_B_joint.npy
最终得到如下所展示的目录结构与文件
Your data/ should be like this:
Team-nebver-valorant
└─data
    ├── train_label.npy
    ├── train_bone_motion.npy
    ├── train_bone.npy
    ├── train_joint_bone.npy
    ├── train_joint_motion.npy
    ├── train_joint.npy
    ├── test_*_bone_motion.npy
    ├── test_*_bone.npy
    ├── test_*_joint_bone.npy
    ├── test_*_joint_motion.npy
    ├── test_*_joint.npy
    ├── ..........
    ├── zero_label_B.npy
.....

2.TRAIN
注：注意修改配置文件的数据路径，训练时使用config/match下的train_longtail.yaml文件，并根据使用的设备数量在配置文件中修改device的信息
训练命令如下：
python main_logit_adjust.py --config ./config/match/train_longtail.yaml
训练得到的置信度文件pred.py保存在work_dir/match/confidence_B，权重和日志保存在work_dir/match/ctrgcn_longtail,最优权重命名为run-best-最大轮次-步数（最大轮次即以run-best开头命名的最大轮次），日志命名为log.txt

3.EVALUATE
验证命令如下：
python main_logit_adjust.py --config ./config/match/test_joint_B.yaml
得到的置信度文件保存在配置文件confidence_dir对应文件中
