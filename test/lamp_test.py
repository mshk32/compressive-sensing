import matplotlib.pyplot as plt
import cv2
from framework import ImageCS, lamp
from pathlib import Path

def process_image(image_path: Path,
                  output_root: Path,
                  K_ratios: list[float],
                  M_ratios: list[float]) -> tuple[list[float], list[float], list[float]]:
    basename = image_path.stem
    print(f"\n=== Processing {basename}{image_path.suffix} ===")

    out_dir = output_root / basename
    out_dir.mkdir(parents=True, exist_ok=True)

    cr_list: list[float] = []
    psnr_list: list[float] = []
    ssim_list: list[float] = []

    for k in K_ratios:
        for m in M_ratios:
            rec = lamp(str(image_path), K_ratio=k, M_ratio=m)

            # Сохраняем восстановленное изображение
            out_file = out_dir / f"{basename}_K={k:.2f}_M={m:.2f}.png"
            cv2.imwrite(str(out_file), rec.get_Image())

            cr, psnr, ssim = rec.get_CR(), rec.get_PSNR(), rec.get_SSIM()
            print(f" K={k:.2f}, M={m:.2f} → CR={cr:.2f}, SSIM={ssim:.3f}, PSNR={psnr:.2f} dB, saved {out_file.name}")

            cr_list.append(cr)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

    return cr_list, psnr_list, ssim_list


def plot_results(results: dict[str, tuple[list[float], list[float]]],
                 output_root: Path) -> None:
    
    plots_dir = output_root / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Собираем все значения
    all_cr = []
    all_psnr = []
    for cr_list, psnr_list in results.values():
        all_cr.extend(cr_list)
        all_psnr.extend(psnr_list)

    # 1. Scatter всех точек
    plt.figure(figsize=(8, 6))
    plt.scatter(all_cr, all_psnr, alpha=0.6)
    plt.xlabel('CR')
    plt.ylabel('PSNR, dB')
    plt.title('Scatter всех точек PSNR vs CR')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / 'scatter_all.png')
    plt.close()

    # 2. Гистограмма всех PSNR
    plt.figure(figsize=(8, 4))
    plt.hist(all_psnr, bins=20)
    plt.xlabel('PSNR, dB')
    plt.ylabel('Частота')
    plt.title('Гистограмма распределения PSNR')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / 'hist_psnr.png')
    plt.close()

    # 3. Гистограмма всех CR
    plt.figure(figsize=(8, 4))
    plt.hist(all_cr, bins=20)
    plt.xlabel('CR')
    plt.ylabel('Частота')
    plt.title('Гистограмма распределения CR')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / 'hist_cr.png')
    plt.close()

    # 4. Отдельные scatter-графики PSNR vs CR по изображениям
    for name, (cr_list, psnr_list) in results.items():
        plt.figure(figsize=(6, 4))
        plt.scatter(cr_list, psnr_list, alpha=0.7)
        plt.xlabel('CR')
        plt.ylabel('PSNR, dB')
        plt.title(f'{name}: PSNR vs CR (scatter)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / f'{name}_psnr_cr_scatter.png')
        plt.close()


def main():
    input_dir  = Path("../misc")
    output_dir = Path("../images/lamp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сетки параметров
    M = [0.1, 0.2, 0.3, 0.4, 0.5]
    K = [0.15, 0.2, 0.25, 0.3]

    # Собираем все файлы в input_dir
    images = [p for p in input_dir.iterdir() if p.is_file()]

    results: dict[str, tuple[list[float], list[float], list[float]]] = {}

    for img_path in images:
        cr_list, psnr_list, ssim_list = process_image(img_path, output_dir, K, M)
        results[img_path.stem] = (cr_list, psnr_list, ssim_list)

    # Строим и сохраняем графики
    plot_results(results, output_dir)

    return results


if __name__ == "__main__":
    main()
