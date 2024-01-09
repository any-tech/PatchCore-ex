import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# データの初期化
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# プロットの初期化
fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

line, = ax1.plot(x, y)
scat = ax2.scatter([], [])

# アニメーションの更新関数
def update(frame):
    # ax1の更新: 正弦波を時間とともに変化させる
    line.set_ydata(np.sin(x + frame / 10.0))

    # ax2の更新: ランダムな点を散布図に追加
    ax2.clear()
    ax2.scatter(np.random.random(10), np.random.random(10))
    return line, scat

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# 動画ファイルとして保存
ani.save('output_animation.mp4', writer='ffmpeg')

plt.show()
