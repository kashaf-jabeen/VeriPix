import os
import time
import csv
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.contrib.auth import login, logout, authenticate, update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.utils.timesince import timesince

from .gradcam import generate_gradcam_overlay
from .forms import RegisterForm, LoginForm, UserUpdateForm, ProfileUpdateForm
from .models import DetectionLog, UserProfile

# Device & Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "veripix_model.pth"

def load_trained_model():
    if os.path.exists(MODEL_PATH):
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
            
        model.to(device)
        model.eval()
        return model
    return None

model = load_trained_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- VIEWS ---

def index(request):
    return render(request, 'detector/index.html')


def clear_history(request):
    """
    Clears session history and logged user's history if authenticated.
    """
    if 'history_results' in request.session:
        del request.session['history_results']
    
    if request.user.is_authenticated:
        DetectionLog.objects.filter(user=request.user).delete()
    else:
        DetectionLog.objects.filter(user__isnull=True).delete()
        
    return redirect('history')


def delete_log(request, log_id):
    """
    Permanently deletes a scan record from both Session and Database
    """
    if request.method == "POST":
        # 1. Database level deletion
        try:
            log_item = DetectionLog.objects.get(id=log_id)
            log_item.delete()
        except DetectionLog.DoesNotExist:
            pass

        # 2. Session level deletion sync
        if 'history_results' in request.session:
            history = request.session['history_results']
            updated_history = [item for item in history if str(item.get('id', '')) != str(log_id)]
            request.session['history_results'] = updated_history
            request.session.modified = True

        # Check request referrer for smart redirect
        referer = request.META.get('HTTP_REFERER', '')
        if 'dashboard' in referer:
            return redirect('dashboard')
        elif 'history' in referer:
            return redirect('history')
        
        return redirect('history')
        
    return redirect('history')


def detect_image(request):
    if 'history_results' not in request.session:
        request.session['history_results'] = []

    current_history = request.session['history_results']

    if request.method == 'POST':
        files = request.FILES.getlist('images') or request.FILES.getlist('image')
        
        # Limit guest user to 5 scans
        if not request.user.is_authenticated:
            if len(current_history) + len(files) > 5:
                return redirect('/login/?next=/detect/&limit_exceeded=1')

        if files:
            fs = FileSystemStorage()
            new_results = []

            for file in files:
                start_time = time.time()
                filename = fs.save(file.name, file)
                file_url = fs.url(filename)
                filepath = fs.path(filename)

                img = Image.open(filepath).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(device)

                if model is not None:
                    outputs = model(input_tensor)
                    logit_0 = outputs[0][0].item()
                    logit_1 = outputs[0][1].item()
                    logit_diff = logit_0 - logit_1

                    if logit_diff > 40.0:
                        prediction = 'FAKE'
                        verdict_db = 'AI'
                        score = round(min(98.5, 75.0 + (logit_diff - 40.0) * 0.5), 2)
                    else:
                        prediction = 'REAL'
                        verdict_db = 'Real'
                        score = round(min(98.5, 75.0 + (40.0 - logit_diff) * 0.5), 2)

                    gradcam_filename = f"gradcam_{filename}"
                    gradcam_path = os.path.join(fs.location, gradcam_filename)
                    try:
                        generate_gradcam_overlay(model, input_tensor, filepath, gradcam_path)
                        gradcam_url = fs.url(gradcam_filename)
                    except Exception:
                        gradcam_url = file_url
                else:
                    prediction = 'MODEL NOT LOADED'
                    verdict_db = 'Real'
                    score = 0.0
                    gradcam_url = file_url

                proc_time = round(time.time() - start_time, 2)

                # --- DATABASE ENTRY CREATION (FIXED & TRACKED) ---
                log_entry_id = None
                try:
                    db_log = DetectionLog.objects.create(
                        user=request.user if request.user.is_authenticated else None,
                        file_name=file.name,
                        file_type='image',
                        verdict=verdict_db,
                        confidence_score=score
                    )
                    log_entry_id = db_log.id
                except Exception as e:
                    print(f"Database Save Error: {e}")

                # Append to current batch
                new_results.append({
                    'id': str(log_entry_id) if log_entry_id else str(int(time.time() * 1000)),
                    'filename': file.name,
                    'image_url': file_url,
                    'gradcam_url': gradcam_url,
                    'prediction': prediction,
                    'verdict': verdict_db,
                    'confidence': score,
                    'proc_time': proc_time,
                })

            updated_history = new_results + current_history
            request.session['history_results'] = updated_history
            request.session.modified = True

    history = request.session.get('history_results', [])

    context = {
        'results': history,
        'total_count': len(history),
    }
    return render(request, 'detector/result.html', context)


# --- SINGLE RESULT VIEW ---

def result_view(request, log_id):
    log_item = get_object_or_404(DetectionLog, id=log_id)
    
    formatted_item = {
        'id': str(log_item.id),
        'filename': log_item.file_name,
        'image_url': f"/media/{log_item.file_name}",
        'gradcam_url': f"/media/gradcam_{log_item.file_name}",
        'prediction': 'FAKE' if log_item.verdict in ['AI', 'FAKE'] else 'REAL',
        'verdict': log_item.verdict,
        'confidence': log_item.confidence_score,
    }

    context = {
        'results': [formatted_item],
        'total_count': 1,
    }
    return render(request, 'detector/result.html', context)


# --- AUTH VIEWS ---

def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('/detect/')
    else:
        form = RegisterForm()
    return render(request, 'detector/register.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            next_url = request.GET.get('next', '/detect/')
            return redirect(next_url)
    else:
        form = LoginForm()
    
    limit_exceeded = request.GET.get('limit_exceeded', False)
    return render(request, 'detector/login.html', {'form': form, 'limit_exceeded': limit_exceeded})


def logout_view(request):
    logout(request)
    return redirect('/')


# --- LIVE DASHBOARD VIEW ---

def dashboard_view(request):
    """
    Fetches live statistics and latest scan records directly from DB.
    """
    if request.user.is_authenticated:
        user_logs = DetectionLog.objects.filter(user=request.user)
    else:
        user_logs = DetectionLog.objects.all()

    total_scans = user_logs.count()
    real_count = user_logs.filter(verdict__iexact='Real').count()
    ai_count = user_logs.filter(verdict__in=['AI', 'FAKE', 'Fake']).count()
    
    real_percentage = round((real_count / total_scans * 100), 1) if total_scans > 0 else 0
    ai_percentage = round((ai_count / total_scans * 100), 1) if total_scans > 0 else 0

    recent_logs = user_logs.order_by('-created_at')[:10]
    model_name = "ResNet-18"

    context = {
        'total_scans': total_scans,
        'real_count': real_count,
        'ai_count': ai_count,
        'real_percentage': real_percentage,
        'ai_percentage': ai_percentage,
        'model_name': model_name,
        'recent_logs': recent_logs,
    }
    return render(request, 'detector/dashboard.html', context)


# --- HISTORY VIEW (LIVE DATABASE TRACKING) ---

def history_view(request):
    """
    Fetches live statistics and all historical logs directly from DB.
    """
    if request.user.is_authenticated:
        logs = DetectionLog.objects.filter(user=request.user).order_by('-created_at')
    else:
        logs = DetectionLog.objects.all().order_by('-created_at')

    total_scans = logs.count()
    fake_count = logs.filter(verdict__in=['AI', 'FAKE', 'Fake']).count()
    real_count = logs.filter(verdict__in=['Real', 'REAL', 'Authentic']).count()
    
    # Dynamic Live Average Confidence Score
    if total_scans > 0:
        total_score = sum([log.confidence_score for log in logs if log.confidence_score is not None])
        avg_confidence = round(total_score / total_scans, 1)
    else:
        avg_confidence = 0.0

    context = {
        'logs': logs,
        'total_scans': total_scans,
        'fake_count': fake_count,
        'real_count': real_count,
        'avg_confidence': avg_confidence,
    }
    return render(request, 'detector/history.html', context)


# --- LIVE HISTORY API ENDPOINT (MATCHED TIMESTAMP WITH DASHBOARD) ---

def history_api(request):
    if request.user.is_authenticated:
        logs = DetectionLog.objects.filter(user=request.user).order_by('-created_at')
    else:
        logs = DetectionLog.objects.all().order_by('-created_at')

    total_scans = logs.count()
    fake_count = logs.filter(verdict__in=['AI', 'FAKE', 'Fake']).count()
    real_count = logs.filter(verdict__in=['Real', 'REAL', 'Authentic']).count()
    
    if total_scans > 0:
        total_score = sum([log.confidence_score for log in logs if log.confidence_score is not None])
        avg_confidence = round(total_score / total_scans, 1)
    else:
        avg_confidence = 0.0

    logs_data = []
    for log in logs:
        # Match Dashboard relative time format (e.g. "2 minutes ago", "1 hour, 7 minutes ago")
        time_str = f"{timesince(log.created_at)} ago" if log.created_at else ''

        logs_data.append({
            'id': log.id,
            'file_type': log.file_type.upper() if log.file_type else 'IMAGE',
            'file_name': log.file_name,
            'verdict': log.verdict,
            'confidence_score': log.confidence_score,
            'created_at': time_str,
        })

    return JsonResponse({
        'logs': logs_data,
        'total_scans': total_scans,
        'fake_count': fake_count,
        'real_count': real_count,
        'avg_confidence': avg_confidence
    })


# --- EXPORT CSV FUNCTIONALITY ---

def export_csv(request):
    """
    Generates and downloads a CSV file containing all scan records.
    """
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="scan_history.csv"'

    writer = csv.writer(response)
    writer.writerow(['Type', 'File Name', 'Verdict', 'Confidence Score', 'Timestamp'])

    if request.user.is_authenticated:
        logs = DetectionLog.objects.filter(user=request.user).order_by('-created_at')
    else:
        logs = DetectionLog.objects.all().order_by('-created_at')

    for log in logs:
        writer.writerow([
            log.file_type.upper() if log.file_type else 'IMAGE',
            log.file_name,
            log.verdict,
            f"{log.confidence_score}%",
            log.created_at.strftime("%Y-%m-%d %H:%M:%S") if log.created_at else ''
        ])

    return response


# --- HEATMAP DETAIL VIEW ---

def heatmap_detail_view(request, log_id):
    log = get_object_or_404(DetectionLog, id=log_id)
    
    context = {
        'log': log,
        'image_url': f"/media/{log.file_name}",
        'gradcam_url': f"/media/gradcam_{log.file_name}"
    }
    return render(request, 'detector/heatmap_detail.html', context)


# --- SETTINGS VIEW WITH REAL CRUD OPERATIONS ---
# python
from django.contrib.auth import update_session_auth_hash

# python
# python
@login_required
def settings_view(request):
    profile, created = UserProfile.objects.get_or_create(user=request.user)

    if request.method == 'POST':
        action = request.POST.get('action')

        # 1. Update Profile
        if action == 'save_profile':
            username = request.POST.get('username')
            email = request.POST.get('email')

            if username:
                request.user.username = username
            if email:
                request.user.email = email
            request.user.save()

            # Fix: Checking request.FILES properly
            if 'profile_pic' in request.FILES and request.FILES['profile_pic']:
                profile.profile_pic = request.FILES['profile_pic']
                profile.save()

            messages.success(request, 'Profile updated successfully!')
            return redirect('settings')

        # 2. Change Password
        elif action == 'change_password':
            curr_pass = request.POST.get('current_password')
            new_pass = request.POST.get('new_password')

            if request.user.check_password(curr_pass):
                request.user.set_password(new_pass)
                request.user.save()
                update_session_auth_hash(request, request.user)
                messages.success(request, 'Password updated successfully!')
            else:
                messages.error(request, 'Current password is incorrect!')

        # 3. DELETE PROFILE PICTURE (FIXED)
        elif action == 'delete_pic':
            if profile.profile_pic and profile.profile_pic.name != 'profile_pics/default.png':
                profile.profile_pic.delete(save=False) # File system se delete
            profile.profile_pic = 'profile_pics/default.png' # Default path reset
            profile.save()
            messages.info(request, 'Profile picture removed successfully.')

        return redirect('settings')

    return render(request, 'detector/settings.html', {'profile': profile})