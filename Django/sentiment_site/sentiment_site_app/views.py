from django.shortcuts import render
from django.http import HttpResponse

# Create views here. Each view is a website element

def index(request):
    return HttpResponse("Hello World!")
