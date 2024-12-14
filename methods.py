import plane
import heapq


def random(model):
    """Creates one boarding group"""
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
    for x in range(18, 2, -1):
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
        for x in range(3, 19):
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
        for x in range(3, 19):
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


def map_shortest_path_boarding(model):
    # Creating an adjacency table for a graph
    graph = {}
    for row in range(21):  # including corridors
        for col in range(7):
            graph[(row, col)] = []
            if col > 0:  # ledt side
                graph[(row, col)].append((row, col - 1))
            if col < 6:  # right side
                graph[(row, col)].append((row, col + 1))
            if row > 0:  # front
                graph[(row, col)].append((row - 1, col))
            if row < 20:  # back
                graph[(row, col)].append((row + 1, col))

    # Calculate the shortest path using Dijkstra's algorithm
    def dijkstra(graph, start, end):
        distances = {node: float("inf") for node in graph}
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

    # Calculate the shortest path length for each passenger
    passengers = []
    id = 1
    for row in range(3, 19):
        for col in (0, 1, 2, 4, 5, 6):
            passenger = plane.PassengerAgent(id, model, (row, col), 1)
            id += 1
            path_length = dijkstra(
                graph, (0, 3), (row, col)
            )  # Ensure that (0, 3) is in the plot
            passengers.append((passenger, path_length))

    # Sorting passengers by shortest path length
    passengers.sort(key=lambda x: x[1])

    # Add sorted passengers to the boarding queue
    model.boarding_queue = [p[0] for p in passengers]


def dp_boarding(model):
    # Get All Seats
    seats = [(row, col) for row in range(3, 19) for col in (0, 1, 2, 4, 5, 6)]
    num_seats = len(seats)

    # Initialize the DP table
    dp = [float("inf")] * (num_seats + 1)
    dp[0] = 0  # Initialized to 0 when there are no passengers

    # Calculate the distance from each seat to the unique gate
    distances = {
        seat: {"cost": calculate_time((0, 3), seat), "id": idx + 1}
        for idx, seat in enumerate(seats)
    }

    # Sort seats by distance
    sorted_seats = sorted(distances.items(), key=lambda d: d[1]["cost"], reverse=True)

    # Populating the DP table
    for i, seat in enumerate(sorted_seats, start=1):
        dp[i] = dp[i - 1] + distances[seat[0]]["cost"]

    # Reconstructing the optimal solution
    optimal_path = []
    ids = []
    total_time = dp[-1]
    for i in range(num_seats, 0, -1):
        optimal_path.append(sorted_seats[i - 1][0])
        ids.append(sorted_seats[i - 1][1]["id"])

    # Create an instance of PassengerAgent and add it to the boarding queue
    model.boarding_queue = [
        plane.PassengerAgent(id, model=model, seat_pos=seat, group=1)
        for seat, id in zip(optimal_path, ids)
    ]


def interleave_arrays(A, B):
    # Make sure both arrays are the same length
    if len(A) != len(B):
        raise ValueError("Both arrays must have the same length")

    n = len(A)
    C = [None] * (2 * n)  # Create a new array of length 2n

    for i in range(n):
        C[2 * i] = A[i]  # Insert the elements of A at even index positions
        C[2 * i + 1] = B[i]  # Insert the elements of B at odd index positions

    return C


def calculate_time(boarding_point, seat):
    # Calculate the time from gate to seat
    return abs(boarding_point[0] - seat[0]) + abs(boarding_point[1] - seat[1])
