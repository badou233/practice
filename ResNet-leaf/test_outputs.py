# 加载模型状态字典的代码
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 176))
net.load_state_dict(torch.load("net_{}_{}_{}.pth".format(lr, batch_size, num_epochs)))
print("模型状态字典已加载")


# 进行预测并输出测试结果
predictions = []
image_names = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(predicted)):
            label = index_to_label[predicted[i].item()]
            predictions.append(label)
            image_names.append(names[i])

# 保存预测结果到 CSV 文件
result_df = pd.DataFrame({
    'image': image_names,
    'label': predictions
})
result_df.to_csv('测试结果.csv', index=False)