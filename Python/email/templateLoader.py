import os

def get_template_path(path):
    file_path = os.path.join(os.getcwd(), path)
    if not os.path.exists(file_path):
        raise Exception("This is not valid path: " + file_path)
    return file_path

def get_template(path):
    file_path = get_template_path(path)
    return open(file_path).read()

def test():
    file = 'newsletter-template.html'

    template_text = get_template(file).format(
        welcome_message = "Welcome!",
        site_address = "www.google.com",
        name = "Joao das Couves",
        company = "Arvere Inc",
        email = "zeh_das_couves@gmail.com",
        reply_adress = "arvers.inc@gmail.com"
    )
    print(template_text)

test()