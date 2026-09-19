def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
    # 这个函数实现FCOS的标签分配策略！
    
    labels = []       # 存储每个位置的类别标签
    reg_targets = []  # 存储每个位置的回归目标
    
    # 拆分位置点的x、y坐标
    xs, ys = locations[:, 0], locations[:, 1]
    
    # 对每张图像单独处理
    for im_i in range(len(targets)):
        targets_per_im = targets[im_i]  # 当前图像的真实标签
        assert targets_per_im.mode == "xyxy"  # 确保是xyxy格式
        
        # 获取真实框的坐标
        bboxes = targets_per_im.bbox
        # 获取真实框的类别标签
        labels_per_im = targets_per_im.get_field("labels")
        # 获取真实框的面积
        area = targets_per_im.area()
        
        # 1. 计算回归目标：位置点到真实框四边的距离
        # 计算每个位置点到每个真实框四边的距离
        l = xs[:, None] - bboxes[:, 0][None]  # 到左边的距离
        t = ys[:, None] - bboxes[:, 1][None]  # 到上边的距离
        r = bboxes[:, 2][None] - xs[:, None]  # 到右边的距离
        b = bboxes[:, 3][None] - ys[:, None]  # 到下边的距离
        
        # 组合成4个距离：左、上、右、下
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
        
        # 2. 判断位置点是否在真实框内（或中心采样区域内）
        if self.center_sampling_radius > 0:
            # 使用中心采样：只考虑中心区域
            is_in_boxes = self.get_sample_region(
                bboxes,
                self.fpn_strides,
                self.num_points_per_level,
                xs, ys,
                radius=self.center_sampling_radius
            )
        else:
            # 不使用中心采样：点在框内就算正样本
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            # min(dim=2)[0]：取4个距离的最小值
            # > 0：如果4个距离都>0，说明点在框内
        
        # 3. 判断位置点是否在当前FPN层的负责范围内
        # 取4个距离的最大值（表示框的大小）
        max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
        
        # 判断是否在指定的大小范围内
        is_cared_in_the_level = \
            (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
            (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])
        # [0]是最小值，[1]是最大值
        
        # 4. 为每个位置点选择最合适的真实框
        # 创建一个面积矩阵：[位置点数量, 真实框数量]
        locations_to_gt_area = area[None].repeat(len(locations), 1)
        
        # 如果位置点不在框内，面积设为INF
        locations_to_gt_area[is_in_boxes == 0] = INF
        
        # 如果位置点不在当前层的负责范围，面积设为INF
        locations_to_gt_area[is_cared_in_the_level == 0] = INF
        
        # 5. 选择面积最小的真实框
        # min(dim=1)：对每个位置点，找面积最小的真实框
        locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)
        # locations_to_min_area：最小面积值
        # locations_to_gt_inds：对应的真实框索引
        
        # 6. 获取最终的回归目标和标签
        # 根据索引选择回归目标
        reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
        # 根据索引选择标签
        labels_per_im = labels_per_im[locations_to_gt_inds]
        
        # 7. 处理背景（没有匹配到任何真实框的位置）
        # 如果最小面积是INF，说明没有匹配到真实框，标签设为0（背景）
        labels_per_im[locations_to_min_area == INF] = 0
        
        # 8. 保存结果
        labels.append(labels_per_im)
        reg_targets.append(reg_targets_per_im)
    
    return labels, reg_targets