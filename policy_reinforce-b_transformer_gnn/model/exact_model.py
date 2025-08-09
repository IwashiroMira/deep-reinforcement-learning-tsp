import numpy as np
from gurobipy import Model, GRB, quicksum


class ExactModel():
    def __init__(self, num_cities, coords):
        self.n = num_cities
        self.coords = coords
        self.model = Model()
        self.model.setParam("OutputFlag", 0)  # ログ非表示
        self.calculate_distance_matrix()

    def calculate_distance_matrix(self):
        self.distance_matrix = np.linalg.norm(
            self.coords[:, np.newaxis] - self.coords[np.newaxis, :], axis=2
        )
    
    def get_visit_order(self, solution):
        n = solution.shape[0]
        visited = [0]            # 出発都市（通常 0）を最初に入れる
        current = 0              # 現在の都市

        for _ in range(n - 1):   # 残りの都市数だけ繰り返す
            next_city = int(np.argmax(solution[current]))  # 今の行から「1」になっている列を探す
            visited.append(next_city)  # その都市を訪問順に追加
            current = next_city        # 現在地を更新

        return visited

    def execute(self):
        n = self.n
        model = self.model
        dist = self.distance_matrix

        # 変数定義
        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
        u = model.addVars(n, vtype=GRB.INTEGER, name="u")

        # 出発制約
        for i in range(n):
            model.addConstr(quicksum(x[i, j] for j in range(n) if i != j) == 1)

        # 到着制約
        for j in range(n):
            model.addConstr(quicksum(x[i, j] for i in range(n) if i != j) == 1)

        # MTZ部分巡回路除去制約
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)

        # 目的関数
        model.setObjective(
            quicksum(dist[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j),
            GRB.MINIMIZE
        )

        # 最適化
        model.optimize()

        # 解の抽出
        total_distance = 0
        solution = np.zeros((n, n))
        if model.status == GRB.OPTIMAL:
            for i in range(n):
                for j in range(n):
                    if x[i, j].X >= 0.99:
                        solution[i, j] = 1
                        total_distance += dist[i][j]
            visit_order = self.get_visit_order(solution)
            print("Gurobi found optimal solution.")
            self.total_distance = total_distance
            self.visit_order = visit_order
        else:
            print("Gurobi failed to find optimal solution.")
            self.total_distance = None
            self.visit_order = None