# python3.7+
import os
import sys
module_path = os.path.abspath(os.path.join(''))
from trex import *


if __name__ == '__main__':
    engine_json_path = sys.argv[1]
    engine_json_name = os.path.basename(engine_json_path)
    assert engine_json_name is not None and engine_json_name[-5:] == '.json', "Not valid {} file".format(engine_json_name)

    set_wide_display()
    plan = EnginePlan(engine_json_path)

    df = plan.df
    print(df['Name'])

    graph = to_dot(plan, layer_type_formatter)
    svg_name = render_dot(graph, engine_json_name[:-5], 'svg')
