t_0, t_f = torch.tensor([t_0]), torch.tensor([t_f])
forward_model = PINN(layers, t_0, t_f)
params = list(forward_model.dnn.parameters())
optimizer = torch.optim.LBFGS(params, 0.001,
                              max_iter = steps,
                              max_eval = None,
                              tolerance_grad = 1e-20,
                              tolerance_change = 1e-20,
                              history_size = 10,
                              line_search_fn = 'strong_wolfe')
def closure():
    optimizer.zero_grad()
    loss, loss_data, loss_ode, _ = forward_model.loss(t_0, t_f, x_0, x_f, t)
    loss_arr.append(loss.item())
    loss.backward()
    forward_model.iter+=1
    if forward_model.iter % 100 == 0:
        rollout = forward_model.dnn(t).clone().detach().numpy()
        plt.clf()
        plt.plot(t.clone().detach().numpy(), rollout[:, 0], label = 'dim1')
        plt.plot(t.clone().detach().numpy(), rollout[:, 1], label = 'dim2')
        plt.plot(t.clone().detach().numpy(), rollout[:, 2], label = 'dim3')
        plt.savefig('dynamics.png')
        print(
            'epoch: %f, total loss: %.5f, data loss: %.5f, pde loss: %.5f' %
            (
                forward_model.iter,
                loss, 
                loss_data, 
                loss_ode
            )
        )
    return loss
start_time = time.time()
optimizer.step(closure)



batch_size = 64  # Set the batch size

# Create a DataLoader to batch the input t
t_dataset = data.TensorDataset(t)
t_loader = data.DataLoader(t_dataset, batch_size=batch_size, shuffle=True)

t_0, t_f = torch.tensor([t_0]), torch.tensor([t_f])
forward_model = PINN(layers, t_0, t_f)
params = list(forward_model.dnn.parameters())
optimizer = optim.Adam(params, lr=lr, eps=1e-08, betas=(0.8, 0.99), weight_decay=0, amsgrad=False)

# Training loop with batched input t
for i in range(steps):
    for batch in t_loader:
        batch_t = batch[0]  # Extract batched t from DataLoader
        optimizer.zero_grad()
        loss, loss_data, loss_ode, u_t = forward_model.loss(t_0, t_f, x_0, x_f, batch_t)
        loss.backward()
        optimizer.step()

    forward_model.iter += 1
    
    if forward_model.iter % 100 == 0:
        # Extract and visualize predictions for the entire time series (detach for plotting)
        rollout = forward_model.dnn(t).clone().detach().numpy()
        u_sum = np.array([u.detach().numpy().sum() for u in u_t]).mean()

        # Clear the plot, plot each dimension of the rollout, and save the figure
        plt.clf()
        plt.plot(t.clone().detach().numpy(), rollout[:, 0], label='dim1')
        plt.plot(t.clone().detach().numpy(), rollout[:, 1], label='dim2')
        plt.plot(t.clone().detach().numpy(), rollout[:, 2], label='dim3')
        plt.legend()
        plt.savefig('dynamics.png')

        # Print loss metrics
        print(
            'epoch: %f, total loss: %.5f, data loss: %.5f, pde loss: %.5f, avg. d_sum: %5f' %
            (
                forward_model.iter,
                loss.item(),
                loss_data.item(),
                loss_ode.item(),
                u_sum
            )
        )