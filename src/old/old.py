    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    step = 0
    all_loss = np.zeros(end_iter, float)
    all_accuracy = np.zeros(end_iter, float)
    reinit = False

    # Iterative Magnitude Pruning main loop
    for _ite in range(start_iter, ITERATION):
        # except for the first iteration, cuz we don't prune in the first
        # iteration
        if not _ite == 0:
            pruning.prune_by_percentile()
            # if the reinit option is activated
            if reinit:
                model.apply(pruning.weight_init)
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy()
                                                      * pruning.mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                # todo: why do we need an initialization here?
                pruning.original_initialization(initial_state_dict)
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        print(f"\n--- Pruning Level [{_ite}/{ITERATION}]: ---")
        logger.debug(f"[{_ite}/{ITERATION}] " + "IMP loop")

        ###################
        # model = network.set_model()

        # torch.save(model, MODEL_DIR + arch_type + str(_ite) + '-model.pt')
        # logger.debug('model is saved...!')
        ###################

        # Print the table of Nonzeros in each layer
        comp_level = utils.print_nonzeros(model)
        comp[_ite] = comp_level
        pbar = tqdm(range(end_iter))

        # Training the network
        for iter_ in pbar:
            logger.debug(f"{iter_}/{end_iter}" + " inside training loop " + arch_type)

            # Test and save the most accurate model
            if iter_ % valid_freq == 0:
                logger.debug("Testing...")
                accuracy = test(model, test_dataloader, criterion, device)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.save_model(model, MODEL_DIR, f"{_ite}_model_{prune_type}.pth.tar")

            # Training
            acc, loss = train(model, train_dataloader, criterion, optimizer,
                              epochs=num_epochs, device=device)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy

            # Frequency for Printing Accuracy and Loss
            if iter_ % print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{end_iter} \n'
                    f'Loss: {loss:.6f} Accuracy: {accuracy:.2f}% \n'
                    f'Best Accuracy: {best_accuracy:.2f}%\n')

        # Calculate the connectivity
        activations = Activations(model, test_dataloader, device, batch_size)
        corr.append(activations.get_correlation())
        # Save the activations
        pickle.dump(corr, open(OUTPUT_DIR + arch_type + "_correlation.pkl", "wb"))

        # save the best model
        # writer.add_scalar('Accuracy/test', best_accuracy, comp_level)
        bestacc[_ite] = best_accuracy

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        # NOTE Loss is computed for every iteration while Accuracy is computed
        # only for every {args.valid_freq} iterations. Therefore Accuracy saved
        # is constant during the uncomputed iterations.
        # NOTE Normalized the accuracy to [0,100] for ease of plotting.
        # Dump Plot values

        # Dumping mask

        # resetting variables into 0
        best_accuracy = 0
        all_loss = np.zeros(end_iter, float)
        all_accuracy = np.zeros(end_iter, float)

    # Dumping Values for Plotting
    comp.dump(OUTPUT_DIR + f"{prune_type}_compression.dat")
    bestacc.dump(OUTPUT_DIR + f"{prune_type}_bestaccuracy.dat")

