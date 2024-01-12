from OmniEvent.infer import infer

text = "2022年北京市举办了冬奥会"

results = infer(text=text, task="EE")

print(results[0]["events"])
