# Earthquake Magnitude Prediction

A sophisticated deep learning system for predicting earthquake magnitude from seismic waveforms using dual-branch neural networks and advanced signal processing techniques.

##  Abstract

 Accurate earthquake magnitude prediction is crucial for seismic monitoring, early warning, and disaster assessment, directly affecting emergency response decisions and public safety protection. Traditional magnitude estimation methods mainly rely on empirical formulas and simple statistical features, such as single parameters like maximum amplitude and dominant frequency, which have obvious limitations when processing complex seismic waveforms: they cannot fully utilize the complex time-frequency information of seismic waveforms, especially around the first-arrival P-waves, and cannot effectively model the feature differences of earthquakes with different magnitudes across multiple time scales. This paper proposes an end-to-end deep learning method called MagNet based on multi-scale feature extraction and time-frequency fusion, which simultaneously processes time-domain waveforms and frequency-domain spectrograms of P-waves through a dual-branch network architecture. The time-domain branch adopts parallel multi-scale convolution modules to extract features at different time scales; the frequency-domain branch utilizes hierarchical 2D convolutional networks and adaptive spectral attention mechanisms to automatically identify magnitude-related time-frequency components, avoiding the complex manual feature engineering steps in traditional methods. Experiments on the large-scale STEAD earthquake dataset show that this method achieves significant improvements over baseline models such as LSTM on key metrics: mean absolute error reduced by 44\% to 0.33, coefficient of determination R² improved to 0.828, with 77.8\% achieving acceptable precision (local magnitude error $\le$ 0.5). This method provides an effective technical approach for real-time seismic monitoring and rapid magnitude estimation, with important scientific value and application prospects.
##  Quick Start

### Prerequisites(environment.yml)

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install h5py tqdm
pip install scipy
```

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stanford Earthquake Dataset (STEAD) team
- PyTorch community for the deep learning framework
- Seismology research community for domain expertise

## Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the development team(2209187058@qq.com).

---

**Note**: This system is designed for research purposes. For operational earthquake monitoring, please consult with seismology experts and follow established protocols.