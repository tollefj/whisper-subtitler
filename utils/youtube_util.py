import youtube_dl


def download_video(url, output="store/video.mp4"):
    ydl_opts = {
        # never download anything above 1200 in height, typically 1080p
        "format": "bestvideo[height<=1200][ext=mp4]+bestaudio[ext=m4a]/mp4",
        "outtmpl": output,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])