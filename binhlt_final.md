II.2.2 An Ill-conditioned Problem

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Hàm $f$ là hàm bậc 2 lồi, có 1 điểm cực tiểu tại $(0, 0)$


\(\nabla f(x) = (0.2 x_1,\ 4 x_2)^T\)

Ta thấy gradient theo hướng $x_2$ nhanh hơn gấp 20 lần so với $x_1$.



Để minh họa, chúng ta sẽ thử áp dụng Gradient Descent cho hàm này với learning rate = 0.4
$x_1$

## CODE

Quát sát hình trên ta thấy, với Gradient Descent với learning rate = 0.4, gradient theo hướng $x_2$ lớn hơn và thay đổi nhanh hơn nhiều so với chiều $x_1$. Do đó, ta chỉ có 2 lựa chọn:
    - Với learning rate nhỏ, ta đảm bảo $x_2$ không bị phân kì, nhưng $x_1$ sẽ hội tụ rất chậm
    - Ngược lại, nếu dùng learning rate lớn, ta tiến nhanh theo hướng $x_1$ nhưng lại bị phân kỳ theo hướng $x_2$ ​

## Minh họa learning rate = 0.2
Ta thấy Với learning rate 0.2, ta đảm bảo $x_2$ không bị phân kì, nhưng $x_1$ sẽ hội tụ rất chậm

## Minh họa learning rate = 0.6

Nếu dùng learning rate = 0.6, ta tiến nhanh theo hướng $x_1$ nhưng lại bị phân kỳ theo hướng $x_2$ ​

Kết luận: Gradient Descent gặp khó khăn khi giải quyết bài toán  Ill-conditioned Problem



## II.2.1 Leaky Averages

Nhắc lại kiến thức về Gradient Descent và SGD Minibatch
Theo Gradient Descent, ta có:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$$

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

Trong đó:
- \( \mathbf{w} \): tham số mô hình
- \( \eta_t \): Learning rate
- \( g_t \): Gradient của hàm loss function tại thời điểm t
- \( \mathbf{x}_t \): Dữ liệu đầu vào
- \( f(\mathbf{x}_{t}, \mathbf{w}) \): hàm loss function khi dùng tham số w để dự đoán trên mẫu \( x_t \)
- \( \partial_{\mathbf{w}} \) : Đạo hàm riêng theo tham số w

Minibatch SGD sẽ giúp tăng tốc độ tính toán, công thức: 


$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

Trong đó:
- \( \mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1}) \): gradient của mẫu \(\mathbf{i}\), tính tại tham số bước \( \mathbf{t-1}  \)
- \( \mathbf{w}_t \): Vector tham số mô hình tại bước cập nhật t−1
- \(\ |\mathcal{B}_t| \): kích thước mini-batch tại bước t
- \( f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) \): Loss function của mẫu dữ liệu thứ i trong batch, khi mô hình dùng tham số \(\mathbf{w}_{t-1}\)
- \(\mathbf{g}_{t, t-1}\): gradient trung bình của mini-batch

Chúng ta thấy rằng, Minibatch SGD không tính gradient trên toàn bộ dataset hay trên một mẫu đơn, ta tính gradient trên một tập con nhỏ (mini-batch) \(\mathcal{B}_t\). Điều này giúp:
- Cho tốc độ và ổn định trung bình tốt hơn.
- Gradient tính trên một mẫu đơn có nhiễu (noisy). Lấy trung bình gradient của nhiều mẫu (mini-batch) sẽ giảm phương sai ước lượng gradient, làm cập nhật ổn định hơn.

Tuy nhiên, nếu chỉ dùng \(\mathbf{g}_{t, t-1}\) (trung bình trên batch hiện tại), ta chỉ tận dụng giảm phương sai trong cùng batch đó. Tuy nhiên, gradient giữa các batch kế tiếp vẫn có nhiễu. Để khắc phục tình trạng trên, chúng ta dùng: "Leaky average"(trung bình rò rỉ)

Ý tưởng "Leaky average" là dùng \(\mathbf{v}_{t}\)  lưu trữ thông tin gradient “tích lũy” từ các bước trước

Công thức:
$$ \mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t,t-1} $$

Trong đó:
- \(\mathcal{v}\) được gọi là vận tốc, 
- \(\mathcal{B}\): hệ số ghi nhớ, \(0 \le \beta < 1\)
- \(\mathcal{B} = 0\), \(\mathbf{v}_t\) bằng gradient hiện tại, không lưu thông tin gradient cũ
- Nếu \(\mathcal{B}\) tiến về 1, \(\mathbf{v}_t\) giữ nhiều thông tin từ các gradient trước đó, do đó mượt hơn và có phương sai nhỏ hơn, giúp giảm nhiễu. Nhưng nếu \(\mathcal{B}\) quá lớn, \(\mathbf{v}_t\) có thể bị đổi hướng rất chậm.

Ta có:
- \( \mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t,t-1} \)
- \( \mathbf{v}_{t-1} = \beta \mathbf{v}_{t-2} + \mathbf{g}_{t-1,t-2} \)

Do đó

\(\mathbf{v}_t = \beta(\beta \mathbf{v}_{t-2} + \mathbf{g}_{t-1,t-2}) + \mathbf{g}_{t,t-1} \\= \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1,t-2} + \mathbf{g}_{t,t-1} 
\\= \beta^{3}\mathbf{v}_{t-3} + \beta^{2}\mathbf{g}_{t-2,\,t-3} + \beta\,\mathbf{g}_{t-1,\,t-2} +\mathbf{g}_{t,\,t-1} \)

Khai triễn k bước thì ta có:
\[\mathbf{v}_t= \beta^{k}\mathbf{v}_{t-k}+ \sum_{\tau = 0}^{k-1} \beta^{\tau}\,\mathbf{g}_{t-\tau,\,t-\tau-1}
\]

Khai triễn t bước thì ta có:
\[\mathbf{v}_t= \beta^{t}\mathbf{v}_{t-t}+ \sum_{\tau = 0}^{t-1} \beta^{\tau}\,\mathbf{g}_{t-\tau,\,t-\tau-1} 
\\=\beta^{t}\mathbf{v}_{0}+ \sum_{\tau = 0}^{t-1} \beta^{\tau}\,\mathbf{g}_{t-\tau,\,t-\tau-1} \]

Khởi tạo \(\mathbf{v}_0 = 0\), ta có:
\[\mathbf{v}_t= \sum_{\tau = 0}^{t-1} \beta^{\tau}\,\mathbf{g}_{t-\tau,\,t-\tau-1}
\\=  \mathbf{g}_{t,t-1} + \beta\,\mathbf{g}_{t-1,t-2} + \beta^{2}\mathbf{g}_{t-2,t-3} + \cdots \]

Với \(0 \le \beta < 1\), Ta có:
\[1 + \beta + \beta^2 + \cdots = \frac{1}{1 - \beta}\]

Vì tổng trọng số lớn hơn 1, nên gọi là "leak