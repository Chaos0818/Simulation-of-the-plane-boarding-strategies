import plane

def random(model):
    """ Creates one boarding group """
    id = 1
    group = []
    for x in range(3, 19):
        for y in (0, 1, 2, 4, 5, 6):
            agent = plane.PassengerAgent(id, model, (x, y), 1)
            id += 1
            group.append(agent)
    model.random.shuffle(group)
    model.boarding_queue.extend(group)


def front_to_back_gr(model):
    final_group = []
    id = 1
    sub_group = []
    for x in range(18, 14, -1):
        for y in (0, 1, 2, 4, 5, 6):
            agent = plane.PassengerAgent(id, model, (x, y), 4)
            id += 1
            sub_group.append(agent)
    model.random.shuffle(sub_group)
    for a in sub_group:
        final_group.append(a)
    sub_group = []
    for x in range(14, 10, -1):
        for y in (0, 1, 2, 4, 5, 6):
            agent = plane.PassengerAgent(id, model, (x, y), 3)
            id += 1
            sub_group.append(agent)
    model.random.shuffle(sub_group)
    for a in sub_group:
        final_group.append(a)
    sub_group = []
    for x in range(10, 6, -1):
        for y in (0, 1, 2, 4, 5, 6):
            agent = plane.PassengerAgent(id, model, (x, y), 2)
            id += 1
            sub_group.append(agent)
    model.random.shuffle(sub_group)
    for a in sub_group:
        final_group.append(a)
    sub_group = []
    for x in range(6, 2, -1):
        for y in (0, 1, 2, 4, 5, 6):
            agent = plane.PassengerAgent(id, model, (x, y), 1)
            id += 1
            sub_group.append(agent)
    model.random.shuffle(sub_group)
    for a in sub_group:
        final_group.append(a)
    model.boarding_queue.extend(final_group)


def back_to_front_gr(model):
    final_group = []
    id = 1
    sub_group = []
    for x in range(6, 2, -1):
        for y in (0, 1, 2, 4, 5, 6):
            agent = plane.PassengerAgent(id, model, (x, y), 4)
            id += 1
            sub_group.append(agent)
    model.random.shuffle(sub_group)
    for a in sub_group:
        final_group.append(a)
    sub_group = []
    for x in range(10, 6, -1):
        for y in (0, 1, 2, 4, 5, 6):
            agent = plane.PassengerAgent(id, model, (x, y), 3)
            id += 1
            sub_group.append(agent)
    model.random.shuffle(sub_group)
    for a in sub_group:
        final_group.append(a)
    sub_group = []
    for x in range(14, 10, -1):
        for y in (0, 1, 2, 4, 5, 6):
            agent = plane.PassengerAgent(id, model, (x, y), 2)
            id += 1
            sub_group.append(agent)
    model.random.shuffle(sub_group)
    for a in sub_group:
        final_group.append(a)
    sub_group = []
    for x in range(18, 14, -1):
        for y in (0, 1, 2, 4, 5, 6):
            agent = plane.PassengerAgent(id, model, (x, y), 1)
            id += 1
            sub_group.append(agent)
    model.random.shuffle(sub_group)
    for a in sub_group:
        final_group.append(a)

    model.boarding_queue.extend(final_group)


def front_to_back(model):

    final_group = []
    group_id = 16
    id = 1
    for x in range(18,2,-1):
        sub_group = []
        for y in (0, 1, 2, 4, 5, 6):
            agent = plane.PassengerAgent(id, model, (x, y), group_id)
            id += 1
            sub_group.append(agent)
        model.random.shuffle(sub_group)
        final_group.extend(sub_group)
        group_id -= 1

    model.boarding_queue.extend(final_group)


def back_to_front(model):

    final_group = []
    group_id = 16
    id = 1
    for x in range(3, 19):
        sub_group = []
        for y in (0, 1, 2, 4, 5, 6):
            agent = plane.PassengerAgent(id, model, (x, y), group_id)
            id += 1
            sub_group.append(agent)
        model.random.shuffle(sub_group)
        final_group.extend(sub_group)
        group_id -= 1

    model.boarding_queue.extend(final_group)


def win_mid_ais(model):

    final_group = []
    id = 1
    sub_group = []
    for y in (2, 4):
        for x in range(3,19):
            agent = plane.PassengerAgent(id, model, (x, y), 3)
            id += 1
            sub_group.append(agent)
    model.random.shuffle(sub_group)
    final_group.extend(sub_group)

    sub_group = []
    for y in (1, 5):
        for x in range(3, 19):
            agent = plane.PassengerAgent(id, model, (x, y), 2)
            id += 1
            sub_group.append(agent)
    model.random.shuffle(sub_group)
    final_group.extend(sub_group)

    sub_group = []
    for y in (0, 6):
        for x in range(3,19):
            agent = plane.PassengerAgent(id, model, (x, y), 1)
            id += 1
            sub_group.append(agent)
    model.random.shuffle(sub_group)
    final_group.extend(sub_group)

    model.boarding_queue.extend(final_group)


def steffen_perfect(model):

    final_group = []
    id = 1
    for y in (2, 4):
        for x in range(3, 19, 2):
            agent = plane.PassengerAgent(id, model, (x, y), 6)
            id += 1
            final_group.append(agent)
    for y in (2, 4):
        for x in range(4, 19, 2):
            agent = plane.PassengerAgent(id, model, (x, y), 5)
            id += 1
            final_group.append(agent)
    for y in (1, 5):
        for x in range(3, 19, 2):
            agent = plane.PassengerAgent(id, model, (x, y), 4)
            id += 1
            final_group.append(agent)
    for y in (1, 5):
        for x in range(4, 19, 2):
            agent = plane.PassengerAgent(id, model, (x, y), 3)
            id += 1
            final_group.append(agent)
    for y in (0, 6):
        for x in range(3, 19, 2):
            agent = plane.PassengerAgent(id, model, (x, y), 2)
            id += 1
            final_group.append(agent)
    for y in (0, 6):
        for x in range(4, 19, 2):
            agent = plane.PassengerAgent(id, model, (x, y), 1)
            id += 1
            final_group.append(agent)

    model.boarding_queue.extend(final_group)


def steffen_modified(model):
    group = []
    id = 1
    for x in range(3, 19, 2):
        for y in (2, 1, 0):
            agent = plane.PassengerAgent(id, model, (x, y), 4)
            id += 1
            group.append(agent)
    model.random.shuffle(group)
    model.boarding_queue.extend(group)
    group = []
    for x in range(3, 19, 2):
        for y in (4, 5, 6):
            agent = plane.PassengerAgent(id, model, (x, y), 3)
            id += 1
            group.append(agent)
    model.random.shuffle(group)
    model.boarding_queue.extend(group)
    group = []
    for x in range(4, 19, 2):
        for y in (2, 1, 0):
            agent = plane.PassengerAgent(id, model, (x, y), 2)
            id += 1
            group.append(agent)
    model.random.shuffle(group)
    model.boarding_queue.extend(group)
    group = []
    for x in range(4, 19, 2):
        for y in (4, 5, 6):
            agent = plane.PassengerAgent(id, model, (x, y), 1)
            id += 1
            group.append(agent)
    model.random.shuffle(group)
    model.boarding_queue.extend(group)

import heapq

def map_shortest_path_boarding(model):
    # 创建图的邻接表
    graph = {}
    for row in range(21):  # 包括走廊
        for col in range(7):
            graph[(row, col)] = []
            if col > 0:  # 向左移动
                graph[(row, col)].append((row, col - 1))
            if col < 6:  # 向右移动
                graph[(row, col)].append((row, col + 1))
            if row > 0:  # 向前移动
                graph[(row, col)].append((row - 1, col))
            if row < 20:  # 向后移动
                graph[(row, col)].append((row + 1, col))

    # 使用Dijkstra算法计算最短路径
    def dijkstra(graph, start, end):
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        queue = [(0, start)]
        while queue:
            current_distance, current_node = heapq.heappop(queue)
            if current_node == end:
                return current_distance
            for neighbor in graph[current_node]:
                distance = current_distance + 1
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(queue, (distance, neighbor))
        return distances[end]

    # 计算每个乘客的最短路径长度
    passengers = []
    id = 1
    for row in range(3, 19):
        for col in (0, 1, 2, 4, 5, 6):
            passenger = plane.PassengerAgent(id, model, (row, col), 1)
            id += 1
            path_length = dijkstra(graph, (0, 3), (row, col))  # 确保(0, 3)在图中
            passengers.append((passenger, path_length))

    # 根据最短路径长度排序乘客
    passengers.sort(key=lambda x: x[1])

    # 将排序后的乘客添加到登机队列
    model.boarding_queue = [p[0] for p in passengers]

def dp_boarding(model):
    # 获取所有座位
    seats = [(row, col) for row in range(3, 19) for col in (0, 1, 2, 4, 5, 6)]
    num_seats = len(seats)
    
    # 初始化DP表
    dp = [float('inf')] * (num_seats + 1)
    dp[0] = 0  # 没有乘客时初始化为0

    # 计算每个座位到唯一登机口的距离
    distances = {seat: {"cost":calculate_time((0, 3), seat), "id":idx+1} for idx, seat in enumerate(seats)}
    
    # 根据距离对座位进行排序
    sorted_seats = sorted(distances.items(), key=lambda d: d[1]["cost"], reverse=True)
    
    # 填充DP表
    for i, seat in enumerate(sorted_seats, start=1):
        dp[i] = dp[i-1] + distances[seat[0]]["cost"]
    
    # 重建最优解
    optimal_path = []
    ids = []
    total_time = dp[-1]
    for i in range(num_seats, 0, -1):
        optimal_path.append(sorted_seats[i-1][0])
        ids.append(sorted_seats[i-1][1]["id"])
    
    # 创建PassengerAgent实例并添加到登机队列
    model.boarding_queue = [plane.PassengerAgent(id, model=model, seat_pos=seat, group=1) for seat, id in zip(optimal_path,ids)]

def interleave_arrays(A, B):
    # 确保两个数组长度相同
    if len(A) != len(B):
        raise ValueError("两个数组必须具有相同的长度")
    
    n = len(A)
    C = [None] * (2 * n)  # 创建一个长度为2n的新数组
    
    for i in range(n):
        C[2 * i] = A[i]     # 在偶数索引位置插入A的元素
        C[2 * i + 1] = B[i] # 在奇数索引位置插入B的元素
    
    return C
def calculate_time(boarding_point, seat):
    # 计算从登机口到座位的时间
    return abs(boarding_point[0] - seat[0]) + abs(boarding_point[1] - seat[1])