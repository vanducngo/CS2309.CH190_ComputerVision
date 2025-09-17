import torch
import torch.nn as nn
import numpy as np
from ..utils import memory, memory_2, memory_2_apq
from .base_adapter import BaseAdapter
from .base_adapter import softmax_entropy
from ..utils.bn_layers import RobustBN1d, RobustBN2d
from ..utils.utils import set_named_submodule, get_named_submodule
from ..utils.custom_transforms import get_tta_transforms
import wandb
from omegaconf import OmegaConf

class RoTTA(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(RoTTA, self).__init__(cfg, model, optimizer)

        # Flags
        self.is_use_improve = cfg.ADAPTER.IMPROVEMENT.IS_USE_IMPROVE 
        self.use_queue = cfg.ADAPTER.IMPROVEMENT.USE_QUEUE
        self.use_adaptive_aging = cfg.ADAPTER.IMPROVEMENT.USE_ADAPTIVE_AGING
        self.use_aging_per_batch = cfg.ADAPTER.IMPROVEMENT.USE_AGING_PER_BATCH
        self.use_aware_score = cfg.ADAPTER.IMPROVEMENT.USE_AWARE_SCORE
        self.base_aging_speed = cfg.ADAPTER.IMPROVEMENT.BASE_AGING_SPEED
        self.lamda_d = cfg.ADAPTER.IMPROVEMENT.LAMBDA_D
        self.age_factor = cfg.ADAPTER.IMPROVEMENT.AGE_FACTOR
        self.wait_for_classes = cfg.ADAPTER.IMPROVEMENT.WAIT_FOR_CLASSES
        self.wait_class_ratio = cfg.ADAPTER.IMPROVEMENT.WAIT_CLASS_RATIO
        self.wait_max_instances = cfg.ADAPTER.IMPROVEMENT.WAIT_MAX_INSTANCES
        
        useImproveStr = f'is_use_improve: {self.is_use_improve}'
        useQueueStr = f'use_queue: {self.use_queue}'
        useAdaptiveAgingStr = f'use_adaptive_aging: {self.use_adaptive_aging}'
        useAgingPerBatchStr = f'use_aging_per_batch: {self.use_aging_per_batch}'
        useAwareScoreStr = f'use_aware_score: {self.use_aware_score}'
        baseAgingSpeedStr = f'base_aging_speed: {self.base_aging_speed}'
        lamdaDStr = f'lamda_d: {self.lamda_d}'
        ageFactorStr = f'age_factor: {self.age_factor}'
        waitForClassStr = f'wait_for_classes: {self.wait_for_classes}'
        waitClassRatio = f'wait_class_ratio: {self.wait_class_ratio}'
        waitMaxInstance = f'wait_max_instances: {self.wait_max_instances}'

        print(f'RoTTA Config:\n-{useImproveStr}\n-{useQueueStr}\n-{useAdaptiveAgingStr}\n-{useAgingPerBatchStr}\n-{useAwareScoreStr}\n-{baseAgingSpeedStr}\n-{lamdaDStr}\n-{ageFactorStr}\n-{waitForClassStr}\n-{waitClassRatio}\n-{waitMaxInstance}')

        if self.wait_for_classes:
            self.is_ready_to_adapt = False
        else:
            self.is_ready_to_adapt = True # Bắt đầu adapt ngay lập tức

        if self.is_use_improve:
            if self.use_queue:
                self.mem = memory_2_apq.CSTU2_APQ(
                    cfg=cfg,
                    capacity=self.cfg.ADAPTER.RoTTA.MEMORY_SIZE, 
                    num_class=cfg.CORRUPTION.NUM_CLASS, 
                    lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T, 
                    lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U
                )
            else:
                self.mem = memory_2.CSTU2(
                    cfg=cfg,
                    capacity=self.cfg.ADAPTER.RoTTA.MEMORY_SIZE, 
                    num_class=cfg.CORRUPTION.NUM_CLASS, 
                    lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T, 
                    lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U
                )
        else:
            self.mem = memory.CSTU(
                capacity=self.cfg.ADAPTER.RoTTA.MEMORY_SIZE, 
                num_class=cfg.CORRUPTION.NUM_CLASS, 
                lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T, 
                lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U
            )
        
        # Khởi tạo một biến để theo dõi entropy
        self.ema_entropy = 0.0
        self.alpha_entropy = 0.99 # Hệ số EMA cho entropy
        
        self.model_ema = self.build_ema(self.model)
        self.transform = get_tta_transforms(cfg)
        self.nu = cfg.ADAPTER.RoTTA.NU
        self.update_frequency = cfg.ADAPTER.RoTTA.UPDATE_FREQUENCY  # actually the same as the size of memory bank
        self.current_instance = 0
        self.last_update_loss = 0.0

        cfg2 = OmegaConf.load("configs/adapter/rotta.yaml")
        wandb.init(
            project="chexpert-rotta-improvement-deep",
            config=OmegaConf.to_container(cfg2, resolve=True), # Log toàn bộ config
            name=f"{cfg.MODEL.ARCH}-adapter{cfg.ADAPTER.NAME}-lr{cfg.ADAPTER.RoTTA.MEMORY_SIZE}-bs{cfg.TEST.BATCH_SIZE}"
        )
        wandb.watch(model, log="all", log_freq=100)

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.memory_processing_time = 0.0 # Lưu thời gian xử lý của batch gần nhất

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # batch data
        with torch.no_grad():
            model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(batch_data)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

        # Tính drift signal (NEW)
        current_batch_entropy = entropy.mean().item()
        if self.ema_entropy == 0.0: # Khởi tạo lần đầu
            self.ema_entropy = current_batch_entropy
        
        drift_signal = 0
        if self.use_adaptive_aging:
            drift_signal = (current_batch_entropy - self.ema_entropy) / (self.ema_entropy + 1e-6)
        
        # Cập nhật EMA entropy
        self.ema_entropy = self.alpha_entropy * self.ema_entropy + (1 - self.alpha_entropy) * current_batch_entropy

        # --- BẮT ĐẦU ĐO THỜI GIAN ---
        self.start_event.record()
        # add into memory
        if self.is_use_improve:
            if self.use_aging_per_batch:
                if self.use_aware_score:
                    current_softmax_dist = self.mem._calculate_softmax_dist()
                else:
                    current_softmax_dist = None

                for i, data in enumerate(batch_data):
                    p_l = pseudo_label[i].item()
                    uncertainty = entropy[i].item()
                    current_instance = (data, p_l, uncertainty)

                    if self.use_aware_score:
                        self.mem.add_instance(current_instance, current_softmax_dist)
                    else:
                        self.mem.add_instance(current_instance)

                    self.current_instance += 1
                    if self.current_instance % self.update_frequency == 0:
                        self.checkForUpdateModel(model, optimizer)
                
                # Add age after all items added in memory
                self.mem.add_age(drift_signal)
            else: 
                for i, data in enumerate(batch_data):
                    p_l = pseudo_label[i].item()
                    uncertainty = entropy[i].item()
                    current_instance = (data, p_l, uncertainty)
                    self.mem.add_instance(current_instance)
                    self.mem.add_age(drift_signal)
                    self.current_instance += 1
                    if self.current_instance % self.update_frequency == 0:
                        self.checkForUpdateModel(model, optimizer)
        else:
            for i, data in enumerate(batch_data):
                p_l = pseudo_label[i].item()
                uncertainty = entropy[i].item()
                current_instance = (data, p_l, uncertainty)
                self.mem.add_instance(current_instance)
                self.mem.add_age()
                self.current_instance += 1
                if self.current_instance % self.update_frequency == 0:
                    self.checkForUpdateModel(model, optimizer)
        

        self.end_event.record()
        torch.cuda.synchronize() # Đợi GPU thực hiện xong tất cả các lệnh trên

        # Lấy kết quả thời gian (tính bằng mili-giây)
        self.memory_processing_time = self.start_event.elapsed_time(self.end_event)

        return ema_out

    def checkForUpdateModel(self, model, optimizer):
        if not self.is_ready_to_adapt:
            # Tính số lớp đã có mẫu
            dist = self.mem.per_class_dist()
            num_classes_with_samples = sum(1 for count in dist if count > 0)
            
            # Kiểm tra điều kiện
            if (num_classes_with_samples / self.mem.num_class) >= self.wait_class_ratio:
                print("Sufficient class diversity reached. Starting adaptation.")
                self.is_ready_to_adapt = True
            elif self.current_instance >= self.wait_max_instances:
                print("Max wait time reached. Starting adaptation.")
                self.is_ready_to_adapt = True

        if self.is_ready_to_adapt:
            self.update_model(model, optimizer)

    def update_model(self, model, optimizer):
        model.train()
        self.model_ema.train()
        # get memory data
        sup_data, ages = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            strong_sup_aug = self.transform(sup_data)
            ema_sup_out = self.model_ema(sup_data)
            stu_sup_out = model(strong_sup_aug)
            instance_weight = timeliness_reweighting(ages)
            l_sup = (softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        l = l_sup
        if l is not None:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            self.last_update_loss = l_sup.item()

        self.update_ema_variables(self.model_ema, self.model, self.nu)

        stats = self.analyze_memory_bank()
        if stats:
            wandb.log(stats, step=self.current_instance)

    def analyze_memory_bank(self):
        if not hasattr(self, 'mem') or self.mem.get_occupancy() == 0:
            print("Memory bank is empty or does not exist. No stats to analyze.")
            return None

        all_ages = []
        all_uncertainties = []

        if self.use_queue:
            for class_heap in self.mem.data:
            # item_tuple có dạng: (-score, item_id, MemoryItem_object)
                for item_tuple in class_heap:
                    # Lấy đối tượng MemoryItem từ vị trí thứ 3 (index 2) của tuple
                    memory_item_obj = item_tuple[2]
                    
                    all_ages.append(memory_item_obj.age)
                    all_uncertainties.append(memory_item_obj.uncertainty)
        else:
            for class_list in self.mem.data:
                for item in class_list:
                    # item -> MemoryItem
                    all_ages.append(item.age)
                    all_uncertainties.append(item.uncertainty)

        if not all_ages:
            print("Memory bank occupancy is non-zero, but no items were found. Skipping analysis.")
            return None

        stats = {
            "Occupancy": self.mem.get_occupancy(),
            "Avg Age": np.mean(all_ages),
            "Max Age": np.max(all_ages),
            "Avg Uncertainty": np.mean(all_uncertainties),
            "Max Uncertainty": np.max(all_uncertainties),
            "Mem Proc Time (ms)": self.memory_processing_time 
        }
        
        for key, value in stats.items():
            if isinstance(value, float):
                stats[key] = round(value, 4)

        return stats

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    def configure_model(self, model: nn.Module):

        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer,
                                self.cfg.ADAPTER.RoTTA.ALPHA)
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)
        return model

def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))