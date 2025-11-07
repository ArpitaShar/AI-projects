# %%
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

# %%
class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

vn = MyVanna(config={'model': 'gpt-oss'})

# %%
vn.connect_to_postgres(host='localhost', dbname='MGNREGA', user='diu_user', password='root', port='5432')

# %%
#vn.connect_to_postgres(host='10.22.0.95', dbname='nrega.applicants', user='dsarmaroy', password='yor^&*34', port='5432')

# %%
vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS where table_catalog='MGNREGA' AND table_schema = 'mg'")

# %%
# The information schema query may need some tweaking depending on your database. This is a good starting point.
df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS where table_catalog='MGNREGA' AND table_schema = 'mg'")

# This will break up the information schema into bite-sized chunks that can be referenced by the LLM
plan = vn.get_training_plan_generic(df_information_schema)
plan

# If you like the plan, then uncomment this and run it to train
#vn.train(plan=plan)

# %%
vn.train(plan=plan)

# %%
training_data = vn.get_training_data()
training_data

# %%
vn.ask("How many columns are there in the mg.mg_demregister_2223 table?")

# %%
# from vanna.flask import VannaFlaskApp
# app = VannaFlaskApp(vn)
# app.run()

# %%
schema_info = vn.run_sql("""
    SELECT table_schema, table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'mg';
""")

vn.train("MGNREGA full schema", schema_info.to_string())

# %%
sql = vn.ask("How is gender distribution in different districts", visualize=True)
print("Generated SQL:",sql)

# %%
sql[2]

# %%
sql = vn.ask("How many different block id are there ?", visualize=True)
print("Generated SQL:",sql)

# result = vn.run_sql(sql)
# print(result)


# %%
#!pip install nbformat


# %%
training_data = vn.get_training_data()
training_data

# %%
from vanna.flask import VannaFlaskApp
app = VannaFlaskApp(vn)
app.run()

# %%
# for id in training_data['id']:
#     print("Removing training data with id:", id)
#     vn.remove_training_data(id=id)

# %%
#model='gpt-oss:latest' created_at='2025-09-01T12:02:59.272968374Z' done=True done_reason='stop' total_duration=23399919939 load_duration=65800020 prompt_eval_count=635 prompt_eval_duration=1059444648 eval_count=1091 eval_duration=22271442672 message=Message(role='assistant', content='```python\nimport pandas as pd\nimport plotly.graph_objects as go\n\n# Convert numeric columns to integers (coerce errors to 0)\nnumeric_cols = [\n    \'women\', \'men_hhp_oth\', \'men_hh_st\', \'men_hh_bpl\',\n    \'men_hh_sc_pers\', \'men_hh_oth_pers\', \'men_hh_oth_cash\',\n    \'men_hh_sc_cash\', \'total_workers\'\n]\nfor col in numeric_cols:\n    df[col] = pd.to_numeric(df[col], errors=\'coerce\').fillna(0).astype(int)\n\nif df.shape[0] == 1:\n    # Single district: use an indicator\n    fig = go.Figure(go.Indicator(\n        mode="number+delta",\n        value=df[\'total_workers\'].iloc[0],\n        title={\'text\': f"Total Workers in {df[\'district_code\'].iloc[0]}"},\n        delta={\'reference\': 0}\n    ))\nelse:\n    # Stacked bar chart for multiple districts\n    fig = go.Figure()\n    categories = [\n        \'women\', \'men_hhp_oth\', \'men_hh_st\', \'men_hh_bpl\',\n        \'men_hh_sc_pers\', \'men_hh_oth_pers\', \'men_hh_oth_cash\',\n        \'men_hh_sc_cash\'\n    ]\n    for cat in categories:\n        fig.add_trace(go.Bar(\n            x=df[\'district_code\'],\n            y=df[cat],\n            name=cat.replace(\'_\', \' \').title()\n        ))\n    fig.update_layout(\n        barmode=\'stack\',\n        title=\'Gender Distribution by District\',\n        xaxis_title=\'District Code\',\n        yaxis_title=\'Number of Workers\',\n        legend_title=\'Gender Category\'\n    )\n\nfig.show()\n```', thinking='The user wants Python plotly code to chart the results of the dataframe. The dataframe df has columns: district_code (object), women (int64), men_hhp_oth (int64), men_hh_st (int64), men_hh_bpl (int64), men_hh_sc_pers (object), men_hh_oth_pers (object), men_hh_oth_cash (object), men_hh_sc_cash (object), total_workers (object). Some columns are object types, but presumably numeric strings maybe. We need to create a plotly chart. It might be a bar chart comparing genders across districts. We need to decide plot type. We can transform numeric columns to int. But the instruction: "Assume the data is in a pandas dataframe called \'df\'. If there is only one value in the dataframe, use an Indicator." They didn\'t say to handle data type conversions. Just produce code.\n\nWe need to check if only one row? We can add logic: if len(df) == 1: create indicator else create bar chart. The bar chart should show district_code on x-axis and maybe stacked bar of categories: women, men_hhp_oth, men_hh_st, men_hh_bpl, men_hh_sc_pers, men_hh_oth_pers, men_hh_oth_cash, men_hh_sc_cash. total_workers can be omitted or used as a separate bar.\n\nPlotly express can do stacked bar.\n\nLet\'s write code:\n\n```\nimport plotly.graph_objects as go\nimport plotly.express as px\nimport pandas as pd\n\n# Convert numeric columns to int if possible\nnumeric_cols = [\'women\', \'men_hhp_oth\', \'men_hh_st\', \'men_hh_bpl\',\n                \'men_hh_sc_pers\', \'men_hh_oth_pers\', \'men_hh_oth_cash\',\n                \'men_hh_sc_cash\', \'total_workers\']\nfor col in numeric_cols:\n    df[col] = pd.to_numeric(df[col], errors=\'coerce\').fillna(0).astype(int)\n\nif df.shape[0] == 1:\n    # Single district: show indicator of total workers and breakdown\n    fig = go.Figure(go.Indicator(\n        mode="number+delta",\n        value=df[\'total_workers\'].iloc[0],\n        title={\'text\': f"Total Workers in {df[\'district_code\'].iloc[0]}"},\n        delta={\'reference\': 0}\n    ))\nelse:\n    # Stacked bar chart\n    fig = go.Figure()\n    categories = [\'women\', \'men_hhp_oth\', \'men_hh_st\', \'men_hh_bpl\',\n                  \'men_hh_sc_pers\', \'men_hh_oth_pers\', \'men_hh_oth_cash\',\n                  \'men_hh_sc_cash\']\n    for cat in categories:\n        fig.add_trace(go.Bar(\n            x=df[\'district_code\'],\n            y=df[cat],\n            name=cat.replace(\'_\', \' \').title()\n        ))\n    fig.update_layout(\n        barmode=\'stack\',\n        title=\'Gender Distribution by District\',\n        xaxis_title=\'District Code\',\n        yaxis_title=\'Number of Workers\',\n        legend_title=\'Gender Category\'\n    )\nfig.show()\n```\n\nThat should satisfy. They said respond with only Python code. No explanation. Use import lines. Provide final code.', images=None, tool_name=None, tool_calls=None)


