def print_goals_and_steps_taken(goals_and_steps):
    final_goals_and_steps = ""
    for goal_step in goals_and_steps:
        final_goals_and_steps += f"{goal_step.user_goal} \n"
        final_goals_and_steps += f"c{"\n".join(goal_step.steps_taken)} \n\n"
    return final_goals_and_steps
