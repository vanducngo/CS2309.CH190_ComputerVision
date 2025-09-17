import math
import torch
import heapq

class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self, aging_speed=1.0):
        self.age += aging_speed

    def get_data(self):
        return self.data, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"

class CSTU2_APQ:
    def __init__(self, cfg, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = self.capacity / self.num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u

        # self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]
        self.data: list[list] = [[] for _ in range(self.num_class)]
        self.item_id_counter = 0
        
        # Flags and values
        self.use_aware_score = cfg.ADAPTER.IMPROVEMENT.USE_AWARE_SCORE
        self.age_factor_bonus = cfg.ADAPTER.IMPROVEMENT.AGE_FACTOR
        self.use_adaptive_aging = cfg.ADAPTER.IMPROVEMENT.USE_ADAPTIVE_AGING
        self.base_aging_speed = cfg.ADAPTER.IMPROVEMENT.BASE_AGING_SPEED
        self.lambda_d = cfg.ADAPTER.IMPROVEMENT.LAMBDA_D
        print(f"Using CSTU2_APQ")

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls)
        return occupancy

    def per_class_dist(self):
        per_class_occupied = [0] * self.num_class
        for cls, class_list in enumerate(self.data):
            per_class_occupied[cls] = len(class_list)

        return per_class_occupied

    def add_instance(self, instance, softmax_dist = None):
        assert (len(instance) == 3)
        x, prediction, uncertainty = instance
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=0)
        new_score = self.heuristic_score(0, uncertainty, prediction, softmax_dist)
        if self.remove_instance(prediction, new_score, softmax_dist):
            new_item_tuple = (-new_score, self.item_id_counter, new_item)
            heapq.heappush(self.data[prediction], new_item_tuple)
            self.item_id_counter += 1

    def remove_instance(self, cls, score, softmax_dist = None):
        class_list = self.data[cls]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            else:
                majority_classes = self.get_majority_classes(softmax_dist)
                return self.remove_from_classes(majority_classes, score, softmax_dist)
        else:
            return self.remove_from_classes([cls], score, softmax_dist)

    def remove_from_classes(self, classes: list[int], score_base, softmax_dist):
        max_class = None
        max_score = -1 # Khởi tạo với giá trị nhỏ

        for cls in classes:
            if not self.data[cls]:
                continue

            # Lấy item tệ nhất của lớp này (nằm ở gốc heap)
            # Do lưu -score, nên score thực là -self.data[cls][0][0]
            current_worst_score_in_class = -self.data[cls][0][0]
            if current_worst_score_in_class > max_score:
                max_score = current_worst_score_in_class
                max_class = cls

        if max_class is not None:
            if max_score > score_base:
                heapq.heappop(self.data[max_class])
                return True
            else:
                return False
        else:
            return True

    def get_majority_classes(self, softmax_dist):
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        classes = []
        for i, occupied in enumerate(per_class_dist):
            if softmax_dist is not None:
                # Có softmax dist => Tức là có sử dụng awareness score => Tìm trên toàn bộ class
                classes.append(i)
            else:
                if occupied == max_occupied:
                    classes.append(i)

        return classes

    def _calculate_softmax_dist(self):
        dist = torch.tensor([len(c) for c in self.data], dtype=torch.float32)
        # Thêm nhiệt độ (temperature) T để kiểm soát độ nhọn của phân phối
        # T > 1 -> mềm hơn, T < 1 -> nhọn hơn
        temperature = 2
        return torch.softmax(dist / temperature, dim=0)

    def heuristic_score(self, age, uncertainty, cls: int = -1, softmax_dist = None):
        timelineness_score = self.lambda_t * 1 / (1 + math.exp(-age / self.capacity))
        uncertainty_score = self.lambda_u * uncertainty / math.log(self.num_class)
        base_score = timelineness_score + uncertainty_score

        if self.use_aware_score and softmax_dist is not None and (cls >= 0):
            penalty = softmax_dist[cls].item()
            base_score += self.lambda_d * penalty

        return base_score

    def add_age(self, drift_signal: float = 0.0):
        aging_speed = self.base_aging_speed
        
        if self.use_adaptive_aging:
            aging_speed += self.age_factor_bonus * max(0, drift_signal)
        
        for cls in range(self.num_class):
            updated_heap_items = []
            for _, item_id, item in self.data[cls]:
                # Increase age
                item.increase_age(aging_speed)
                # Re-calculate score
                new_score = self.heuristic_score(item.age, item.uncertainty, cls, None)
                updated_heap_items.append((-new_score, item_id, item))
            
            # Rebuild heap
            heapq.heapify(updated_heap_items)
            self.data[cls] = updated_heap_items

    def get_memory(self):
        tmp_data = []
        tmp_age = []

        for class_heap in self.data:
            for _, _, item in class_heap:
                tmp_data.append(item.data)
                tmp_age.append(item.age)

        tmp_age = [x / self.capacity for x in tmp_age]

        return tmp_data, tmp_age

