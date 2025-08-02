class EvaluationTasks:
    tasks: list = []

    def append(self, t):
        self.tasks.append(t)

    def get_tasks(self):
        return self.tasks

    def num_tasks(self):
        return len(self.tasks)

    def clear_tasks(self):
        self.tasks.clear()


global_evaluation_tasks = EvaluationTasks()
