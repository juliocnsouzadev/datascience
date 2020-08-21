import os

def get_template_path(path):
    file_path = os.path.join(os.getcwd(), path)
    if not os.path.exists(file_path):
        raise Exception("This is not valid path: " + file_path)
    return file_path

def get_template(path):
    file_path = get_template_path(path)
    return open(file_path).read()

def render_template(template_string, data):
    return template_string.format(**data)

def test():
    file = 'newsletter-template.html'
    data = {
        "welcome_message" : "Welcome!",
        "site_address" : "www.google.com",
        "name" : "Joao das Couves",
        "company" : "Arvere Inc",
        "email" : "zeh_das_couves@gmail.com",
        "reply_adress" : "arvers.inc@gmail.com"
    }
    template_text = get_template(file)
    print(render_template(template_text, data))

#test()