from django.shortcuts import render
from django.http import HttpResponse
import plotting

def home(request):
    plot_url = plotting.generate_plot()
    return render(request, 'home.html', {'plot_url': plot_url})
