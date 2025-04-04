s = []

def loading(msg = "Loading"):
    global s
    s.append(msg)
    print(f'{msg}....\n')
    
def done(op = 1):
    if op:
        print(f"Done {s[-1]}\n")
    s.pop()