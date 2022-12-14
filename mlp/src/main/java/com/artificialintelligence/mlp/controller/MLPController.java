package com.artificialintelligence.mlp.controller;

import com.artificialintelligence.mlp.model.*;
import com.artificialintelligence.mlp.model.FuncoesTransferencia.Linear;
import com.artificialintelligence.mlp.model.FuncoesTransferencia.Logistica;
import com.artificialintelligence.mlp.model.MatrizConfusao;
import com.artificialintelligence.mlp.model.FuncoesTransferencia.TangenteHiperbolica;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.List;

@RestController
public class MLPController {
    MatrizConfusao matrizConfusao;

    CalculationParameters calculationParameters;
    String[] classes;
    double[][] dadosOneHotEncoding;

    double taxaAprendizagem;
    double valorErro;
    int opcaoFuncaoTransferencia;


    NormalizacaoValores[] fatorDeNormalizacao;
    ArrayList<Neuronio> neuronios;
    Pesos pesos;
    List<MediaErroRede> mediaErroRedeTotal;

    @GetMapping("/grafico")
    public List<MediaErroRede> saidaDadosGrafico() {
//        System.out.println("ENTROU - ");
        return mediaErroRedeTotal;
    }

    @GetMapping("/matriz")
    public MatrizConfusao saidaDadosMatriz() {
//        System.out.println("ENTROU - ");
        return matrizConfusao;
    }

    @PostMapping("/matriz")
    public MatrizConfusao calculaArquivoTeste(@RequestBody EntradaDados dados) {
        System.out.println("\n\nCALCULANDO MATRIZ CONFUSÃO");
        opcaoFuncaoTransferencia = dados.getCalculationParameters().getTransferFunction();

        dadosOneHotEncoding = new OneHotEncoding()
                .tratarDados(dados.getTrainingData(), dados.getCalculationParameters().getOutputLayer(),
                        classes, opcaoFuncaoTransferencia);

        int numCamadaOculta = dados.getCalculationParameters().getHiddenLayer();
        int totalLinhas = dadosOneHotEncoding.length;
        int totalColunas = dadosOneHotEncoding[0].length;
        int totalEntradas = totalColunas - classes.length;
        int totalSaidas = classes.length;
        int totalNeuronio = totalEntradas + (totalEntradas * numCamadaOculta) + totalSaidas;
        int posDesejado;

        double valorSaidaArquivo;

        matrizConfusao = new MatrizConfusao(totalSaidas, classes);

        normalizarEntradas(dadosOneHotEncoding, totalEntradas);



        for (int i = 0; i < totalLinhas; i++) {
            // -> Gera neuronios de ENTRADA
            neuronios = new ArrayList<Neuronio>();
            for (int j = 0; j < totalEntradas; j++) {
                neuronios.add(new Neuronio(0, dadosOneHotEncoding[i][j], 0));
            }

            // -> Gera CAMADA OCULTA
            for (int camadaAtual = 0; camadaAtual < numCamadaOculta; camadaAtual++) {
                for (int j = 0; j < totalEntradas; j++) {
                    neuronios.add(new Neuronio(0, 0, 0));
                }
            }

            // -> Gera neuronios de SAIDA
            for (int j = 0; j < totalSaidas; j++) {
                neuronios.add(new Neuronio(0, 0, 0));
            }

            if (opcaoFuncaoTransferencia == 1) {
                calcularNetSaida(neuronios, pesos, totalEntradas, new Linear());
            } else if (opcaoFuncaoTransferencia == 2) {
                calcularNetSaida(neuronios, pesos, totalEntradas, new Logistica());
            } else {
                calcularNetSaida(neuronios, pesos, totalEntradas, new TangenteHiperbolica());
            }

            // --> Verifica quem está mais próximo de 1, que será o resultado
            Neuronio tempNeuronio;
            int posMaior = totalNeuronio - totalSaidas;
            double saidaMenor = neuronios.get(posMaior).getSaida();
            double valorSaidaAtual;
            for(int atual = posMaior+1; atual < totalNeuronio; atual++) {
                valorSaidaAtual = neuronios.get(atual).getSaida();
                if (valorSaidaAtual > saidaMenor) {
                    saidaMenor = valorSaidaAtual;
                    posMaior = atual;
                }
            }

            posDesejado = retornaPosDesejado(i, totalEntradas, totalColunas);

            posMaior =  posMaior - (totalNeuronio - totalSaidas);
            switch (posMaior) {
                case 0:
                    matrizConfusao.setMatriz(posDesejado, posMaior);
                    break;
                case 1:
                    matrizConfusao.setMatriz(posDesejado, posMaior);
                    break;
                case 2:
                    matrizConfusao.setMatriz(posDesejado, posMaior);
                    break;
                case 3:
                    matrizConfusao.setMatriz(posDesejado, posMaior);
                    break;
                case 4:
                    matrizConfusao.setMatriz(posDesejado, posMaior);
                    break;
            }

        }

        // -- TESTE -> exibir Matriz Confução
//        System.out.println("MATRIZ CONFUSÃO");
//        int[][] matriz = matrizConfusao.getMatriz();
//        for (int i=0; i < matriz.length; i++) {
//            for (int j=0; j < matriz.length; j++) {
//                System.out.printf(matriz[i][j] + " ");
//            }
//            System.out.printf("\n");
//        }

        // -- Calcula a Acurácia
        double acerto = 0;
        double erro = 0;
        int[][] matriz = matrizConfusao.getMatriz();
        for (int i=0; i < matriz.length; i++) {
            for (int j=0; j < matriz.length; j++) {
                if(i == j) {
                    acerto += matriz[i][j];
                } else {
                    erro += matriz[i][j];
                }
            }
        }

        double acuracia = acerto / (acerto + erro) * 100;
        System.out.println("ACERTO - " + acerto + " | ERRO - " + erro);
        System.out.println("Acurácia - " + acuracia);
        matrizConfusao.setAcuracia(acuracia);

        return matrizConfusao;
    }

    int retornaPosDesejado(int i, int totalEntradas, int totalColunas) {
        int posDesejado = 0;
        for(int j=totalEntradas; j < totalColunas; j++) {
            if (dadosOneHotEncoding[i][j] == 1) {
                posDesejado = j;
            }
        }
        posDesejado = posDesejado - totalEntradas;
        return posDesejado;
    }

    @PostMapping("/entrada")
    public EntradaDados entradaDados(@RequestBody EntradaDados dados) {


        opcaoFuncaoTransferencia = dados.getCalculationParameters().getTransferFunction();

        // 1º - Transformar a Matriz e faz One Hot Encode
        classes = new OneHotEncoding()
                .retornarClasses(dados.getTrainingData(), dados.getCalculationParameters().getOutputLayer());
        dadosOneHotEncoding = new OneHotEncoding()
                .tratarDados(dados.getTrainingData(), dados.getCalculationParameters().getOutputLayer(),
                        classes, opcaoFuncaoTransferencia);
        // --> Exibição
//        System.out.println("\n --- VALORES");
//        new OneHotEncoding().Exibicao(dadosOneHotEncoding);


        // --- ENTRADA
        boolean sorteioPesos = true;

        pesos = new Pesos();
        mediaErroRedeTotal = new ArrayList<MediaErroRede>();
        int numRepeticoes = 0;
        int totalRepeticoes = dados.getCalculationParameters().getNumberIterations();
        int numCamadaOculta = dados.getCalculationParameters().getHiddenLayer();
        ArrayList<Double> vetorErroRede;
        double mediaErroRedeAtual;
        taxaAprendizagem = dados.getCalculationParameters().getLearningRate();
        valorErro = dados.getCalculationParameters().getErrorValue();
        int totalLinhas = dadosOneHotEncoding.length;
        int totalEntradas = dadosOneHotEncoding[0].length - classes.length;
        int totalSaidas = classes.length;

        // *** NORMALIZAR DADOS DE ENTRADA
        fatorDeNormalizacao = new NormalizacaoValores[totalEntradas];
        normalizarEntradas(dadosOneHotEncoding, totalEntradas);

        System.out.println("\n --- VALORES - NORMALIZADOS");
        new OneHotEncoding().Exibicao(dadosOneHotEncoding);
        System.out.println("\n\n");


        do {
            vetorErroRede  = new ArrayList<Double>();
            for (int i = 0; i < totalLinhas; i++) {
                // -> Gera neuronios de ENTRADA
                neuronios = new ArrayList<Neuronio>();
                for (int j = 0; j < totalEntradas; j++) {
                    neuronios.add(new Neuronio(0, dadosOneHotEncoding[i][j], 0));
                }

                // -> Gera CAMADA OCULTA
                for (int camadaAtual = 0; camadaAtual < numCamadaOculta; camadaAtual++) {
                    for (int j = 0; j < totalEntradas; j++) {
                        neuronios.add(new Neuronio(0, 0, 0));
                    }
                }

                // -> Gera neuronios de SAIDA
                for (int j = 0; j < totalSaidas; j++) {
                    neuronios.add(new Neuronio(0, 0, 0));
                }

                // Aqui entra só uma vez
                if (sorteioPesos) {
                    pesos.inicializarPesos(totalEntradas, totalSaidas, numCamadaOculta,
                            opcaoFuncaoTransferencia);
                    sorteioPesos = false;

                    // EXIBIR PESOS INICIAIS
//                    System.out.println("\n\n - PESOS INICIAIS");
//                    var entradaTodos = pesos.getAllPosEntrada();
//                    var saidaTodos = pesos.getAllPosSaida();
//                    var pesoTodos = pesos.getAllPeso();
//                    int ate = entradaTodos.size();
//                    System.out.println();
//                    for (int l = 0; l < ate; l++) {
//                        System.out.printf("Entrada: %d | Saida: %d\t | Peso: %.2f\n",
//                                entradaTodos.get(l), saidaTodos.get(l), pesoTodos.get(l));
//                    }
                }


                // -- TESTE
                // Exibição dos PESOS --->
//                var entradaTodos = pesos.getAllPosEntrada();
//                var saidaTodos = pesos.getAllPosSaida();
//                var pesoTodos = pesos.getAllPeso();
//                int ate = entradaTodos.size();
//                System.out.println();
//                for (int l = 0; l < ate; l++) {
//                    System.out.printf("Entrada: %d | Saida: %d\t | Peso: %.2f\n",
//                            entradaTodos.get(l), saidaTodos.get(l), pesoTodos.get(l));
//                }
                // ------


                if (opcaoFuncaoTransferencia == 1) {
                    calcularNetSaida(neuronios, pesos, totalEntradas, new Linear());

                    calcularErroSaida(neuronios, totalSaidas,
                            dadosOneHotEncoding[i], new Linear());

                    calcularErroCamadaOculta(neuronios, pesos, new Linear(),
                            totalEntradas, totalSaidas);
                } else if (opcaoFuncaoTransferencia == 2) {
                    calcularNetSaida(neuronios, pesos, totalEntradas, new Logistica());

                    calcularErroSaida(neuronios, totalSaidas,
                            dadosOneHotEncoding[i], new Logistica());

                    calcularErroCamadaOculta(neuronios, pesos, new Logistica(),
                            totalEntradas, totalSaidas);
                } else {
                    calcularNetSaida(neuronios, pesos, totalEntradas, new TangenteHiperbolica());

                    calcularErroSaida(neuronios, totalSaidas,
                            dadosOneHotEncoding[i], new TangenteHiperbolica());

                    calcularErroCamadaOculta(neuronios, pesos, new TangenteHiperbolica(),
                            totalEntradas, totalSaidas);
                }

                vetorErroRede.add(erroRede(neuronios, totalSaidas, dadosOneHotEncoding[i]));

                // *** EXIBIÇÃO
//                Systemm.out.println("MEDIA PARCIAL - " +
//                        vetorErroRede.get(vetorErroRede.size()-1));


                // ** Calcular PESOS das ARESTAS
                atualizaPesos(neuronios, pesos, totalEntradas,
                        taxaAprendizagem);


                // *** TIRAR DEPOIS ***
                // EXIBICAO TESTE
//                int ate = neuronios.size();
//                System.out.println();
//                for (int l = 0; l < ate; l++) {
//                    System.out.printf("Net: %.2f\t | Saida: %.2f\t | Erro: %.4f \n",
//                            neuronios.get(l).getNet(), neuronios.get(l).getSaida(),
//                            neuronios.get(l).getErro());
//                }

            }

            mediaErroRedeAtual = calculaMediaRedeAtual(vetorErroRede);
            if (numRepeticoes % 20 == 0) {
                mediaErroRedeTotal.add(new MediaErroRede(numRepeticoes, mediaErroRedeAtual));
                System.out.printf("MÉDIA ERRO DE REDE ["+(numRepeticoes+1)+"] - "+ mediaErroRedeAtual);
                System.out.println("");
            }


            numRepeticoes++;


        // Coloque 1 para TESTE
//        }  while (numRepeticoes < 1 &&
//                mediaErroRedeAtual > valorErro);
        // -- FORMA ABAIXO É A CERTA
        }  while (numRepeticoes < totalRepeticoes &&
        mediaErroRedeAtual > valorErro);
        System.out.println("\n *** FIM CALCULO MLP ***\n");


        // -- TESTE
        // Exibição dos PESOS --->
//        var entradaTodos = pesos.getAllPosEntrada();
//        var saidaTodos = pesos.getAllPosSaida();
//        var pesoTodos = pesos.getAllPeso();
//        int ate = entradaTodos.size();
//        System.out.println();
//        for (int l = 0; l < ate; l++) {
//            System.out.printf("Entrada: %d | Saida: %d\t | Peso: %.2f\n",
//                    entradaTodos.get(l), saidaTodos.get(l), pesoTodos.get(l));
//        }
        // ------



        return dados;
    }

    public double[] retornaMinMax(double[][] dados, int coluna, int totalLinhas) {

        double[] minMax = new double[2];
        minMax[0] = dados[0][coluna]; // Mínimo
        minMax[1] = dados[0][coluna]; // Máximo
        for(int linha=1; linha < totalLinhas; linha++) {
            // Menor
            if (dados[linha][coluna] < minMax[0]) {
                minMax[0] = dados[linha][coluna];
            }
            // Maior
            if (dados[linha][coluna] > minMax[1]) {
                minMax[1] = dados[linha][coluna];
            }
        }

        return minMax;
    }


    public void normalizarEntradas(double[][] dados, int totalEntradas) {
        int totalLinhas = dados.length;
        double[] minMax;
        for (int coluna=0; coluna < totalEntradas; coluna++) {
            minMax = retornaMinMax(dados, coluna, totalLinhas);
            fatorDeNormalizacao[coluna] = new NormalizacaoValores(minMax[0], minMax[1]);
        }

        // EXIBICAO
//        System.out.println("--- EXIBICAO MIN E MAX NORMALIZAÇÃO ---");
//        for (int i=0; i < totalEntradas; i++) {
//            System.out.println("MIN: "+ fatorDeNormalizacao[i].getMin() +
//                    " e MAX - " + fatorDeNormalizacao[i].getMax());
//        }

        // *** NORMALIZAÇÃO
        for (int coluna=0; coluna < totalEntradas; coluna++) {
            for (int linha=0; linha < totalLinhas; linha++) {
                dados[linha][coluna] = (dados[linha][coluna] - fatorDeNormalizacao[coluna].getMin()) /
                                        (fatorDeNormalizacao[coluna].getMax() -
                                                fatorDeNormalizacao[coluna].getMin());
            }
        }

    }

    public double somatorioPesoSaida(int totalPesos, int posNeuronioAtual) {
        int posNeuronioEntrada;
        double somatorio = 0, tempPeso;
        Neuronio neuronioEntrada;
        int pos = 0;
        while (pos < totalPesos) {
            if (posNeuronioAtual == pesos.getPosSaida(pos)) {
                posNeuronioEntrada = pesos.getPosEntrada(pos);
                neuronioEntrada = neuronios.get(posNeuronioEntrada);
                tempPeso = pesos.getPeso(pos);
                somatorio += neuronioEntrada.getSaida() * tempPeso;
            }
            pos++;
        }

        return somatorio;
    }

    public void calcularNetSaida(ArrayList<Neuronio>  neuronios, Pesos pesos, int totalEntradas,
                                 Linear linear) {
        int totalNeuronios = neuronios.size();
        double somatorio = 0;
        int totalPesos;
        for (int posNeuronioAtual = totalEntradas;
             posNeuronioAtual < totalNeuronios; posNeuronioAtual++) {
            somatorio = 0;
            totalPesos =  pesos.getAllPosSaida().size();

            somatorio = somatorioPesoSaida(totalPesos, posNeuronioAtual);

            neuronios.get(posNeuronioAtual).setNet(somatorio);

            // Calcula SAIDA
            double saidaNeuronio = linear.calcularFuncaoSaida(somatorio);
            neuronios.get(posNeuronioAtual).setSaida(saidaNeuronio);
        }
    }

    public void calcularNetSaida(ArrayList<Neuronio>  neuronios, Pesos pesos, int totalEntradas,
        Logistica logistica) {
        int totalNeuronios = neuronios.size();
        double somatorio = 0;
        int totalPesos;
        for (int posNeuronioAtual = totalEntradas;
             posNeuronioAtual < totalNeuronios; posNeuronioAtual++) {
            somatorio = 0;
            totalPesos =  pesos.getAllPosSaida().size();

            somatorio = somatorioPesoSaida(totalPesos, posNeuronioAtual);

            neuronios.get(posNeuronioAtual).setNet(somatorio);

            // Calcula SAIDA
            double saidaNeuronio = logistica.calcularFuncaoSaida(somatorio);
            neuronios.get(posNeuronioAtual).setSaida(saidaNeuronio);
        }
    }

    public void calcularNetSaida(ArrayList<Neuronio>  neuronios, Pesos pesos, int totalEntradas,
                                 TangenteHiperbolica tangenteHiperbolica) {
        int totalNeuronios = neuronios.size();
        double somatorio = 0;
        int totalPesos;
        for (int posNeuronioAtual = totalEntradas;
             posNeuronioAtual < totalNeuronios; posNeuronioAtual++) {
            somatorio = 0;
            totalPesos =  pesos.getAllPosSaida().size();

            somatorio = somatorioPesoSaida(totalPesos, posNeuronioAtual);

            neuronios.get(posNeuronioAtual).setNet(somatorio);

            // Calcula SAIDA
            double saidaNeuronio = tangenteHiperbolica.calcularFuncaoSaida(somatorio);
            neuronios.get(posNeuronioAtual).setSaida(saidaNeuronio);
        }
    }


    public void calcularErroSaida(ArrayList<Neuronio>  neuronios, int totalSaidas,
                                  double[] dados, Linear linear) {

        double desejado, obtido, erro;
        int totalNeuronios = neuronios.size();
        for(int atual = totalNeuronios - totalSaidas, classe = dados.length - totalSaidas;
            atual < totalNeuronios; atual++, classe++) {
            desejado = dados[classe];
            obtido = neuronios.get(atual).getSaida();
            erro = (desejado - obtido) * linear.derivada();
            neuronios.get(atual).setErro(erro);
        }
    }

    public void calcularErroSaida(ArrayList<Neuronio>  neuronios, int totalSaidas,
                                  double[] dados, Logistica logistica) {

        double desejado, obtido, erro;
        int totalNeuronios = neuronios.size();
        for(int atual = totalNeuronios - totalSaidas, classe = dados.length - totalSaidas;
            atual < totalNeuronios; atual++, classe++) {
            desejado = dados[classe];
            obtido = neuronios.get(atual).getSaida();
            erro = (desejado - obtido) * logistica.derivada(obtido);
            neuronios.get(atual).setErro(erro);
        }
    }

    public void calcularErroSaida(ArrayList<Neuronio>  neuronios, int totalSaidas,
                                  double[] dados, TangenteHiperbolica tangenteHiperbolica) {

        double desejado, obtido, erro;
        int totalNeuronios = neuronios.size();
        for(int atual = totalNeuronios - totalSaidas, classe = dados.length - totalSaidas;
            atual < totalNeuronios; atual++, classe++) {
            desejado = dados[classe];
            obtido = neuronios.get(atual).getSaida();
            erro = (desejado - obtido) * tangenteHiperbolica.derivada(obtido);
            neuronios.get(atual).setErro(erro);
        }
    }

    public double erroRede(ArrayList<Neuronio>  neuronios, int totalSaidas, double[] dados) {
        int totalNeuronios = neuronios.size();
        int inicio = totalNeuronios - totalSaidas;
        double desejado, obtido;
        double erroRede = 0;

        for(int atual = totalNeuronios - totalSaidas, classe = dados.length - totalSaidas;
            atual < totalNeuronios; atual++, classe++) {
            desejado = dados[classe];
            obtido = neuronios.get(atual).getSaida();
            erroRede += Math.pow((desejado - obtido), 2);
        }

        erroRede /= 2.0;
//        System.out.println("Erro Rede - " + erroRede);
        return erroRede;
    }

    public double calculaMediaRedeAtual(ArrayList<Double> erroRede) {
        double tamanho = erroRede.size();
        double somatorioErros = 0;
        for(int i=0; i < tamanho; i++) {
            somatorioErros += erroRede.get(i);
        }

        return somatorioErros / tamanho;
    }

    double somatorioPesosCamadaOculta(ArrayList<Neuronio>  neuronios, Pesos pesos, int posInicial,
                                      int posNeuronioAtual, int totalPesos) {

        int posNeuronioSaida;
        int pos = posInicial;
        double tempPeso, somatorio = 0;
        Neuronio neuronioSaida;
        while (pos < totalPesos) {
            if (posNeuronioAtual == pesos.getPosEntrada(pos)) {
                posNeuronioSaida = pesos.getPosSaida(pos);
                neuronioSaida = neuronios.get(posNeuronioSaida);
                tempPeso = pesos.getPeso(pos);
                somatorio += neuronioSaida.getErro() * tempPeso;
            }
            pos++;
        }

        return somatorio;
    }

    void calcularErroCamadaOculta(ArrayList<Neuronio>  neuronios, Pesos pesos, Linear linear,
                                  int totalEntrada, int totalSaidas) {

        int ateNeuronio = neuronios.size() - totalSaidas;
        int totalPesos = pesos.getAllPeso().size();
        int inicioPesosCamadaOculta = totalEntrada * totalEntrada;
        double somatorio;
        Neuronio neuronioAtual;
        for (int posNeuronioAtual=totalEntrada;
             posNeuronioAtual < ateNeuronio; posNeuronioAtual++) {

            neuronioAtual = neuronios.get(posNeuronioAtual);

            somatorio = somatorioPesosCamadaOculta(neuronios, pesos, inicioPesosCamadaOculta,
                                        posNeuronioAtual, totalPesos);

            somatorio *= linear.derivada();
            neuronioAtual.setErro(somatorio);
        }
    }

    void calcularErroCamadaOculta(ArrayList<Neuronio>  neuronios, Pesos pesos, Logistica logistica,
                                  int totalEntrada, int totalSaidas) {

        int ateNeuronio = neuronios.size() - totalSaidas;
        int totalPesos = pesos.getAllPeso().size();
        int inicioPesosCamadaOculta = totalEntrada * totalEntrada;
        double somatorio;
        Neuronio neuronioAtual;
        for (int posNeuronioAtual=totalEntrada;
             posNeuronioAtual < ateNeuronio; posNeuronioAtual++) {

            neuronioAtual = neuronios.get(posNeuronioAtual);

            somatorio = somatorioPesosCamadaOculta(neuronios, pesos, inicioPesosCamadaOculta,
                    posNeuronioAtual, totalPesos);

            somatorio *= logistica.derivada(neuronioAtual.getSaida());
            neuronioAtual.setErro(somatorio);
        }
    }

    void calcularErroCamadaOculta(ArrayList<Neuronio>  neuronios, Pesos pesos,
                                  TangenteHiperbolica tangenteHiperbolica,
                                  int totalEntrada, int totalSaidas) {

        int ateNeuronio = neuronios.size() - totalSaidas;
        int totalPesos = pesos.getAllPeso().size();
        int inicioPesosCamadaOculta = totalEntrada * totalEntrada;
        double somatorio;
        Neuronio neuronioAtual;
        for (int posNeuronioAtual=totalEntrada;
             posNeuronioAtual < ateNeuronio; posNeuronioAtual++) {

            neuronioAtual = neuronios.get(posNeuronioAtual);

            somatorio = somatorioPesosCamadaOculta(neuronios, pesos, inicioPesosCamadaOculta,
                    posNeuronioAtual, totalPesos);

            somatorio *= tangenteHiperbolica.derivada(neuronioAtual.getSaida());
            neuronioAtual.setErro(somatorio);
        }
    }

    void atualizaPesos(ArrayList<Neuronio>  neuronios, Pesos pesos, int totalEntradas,
                       double taxaAprendizagem) {
        int totalPesos = pesos.getAllPeso().size();
        int pesosEntradas = totalEntradas * totalEntradas;
        int posEntrada, posSaida;
        Neuronio auxNeuronio;
        double pesoAtual, novoPeso;
        // Quando as entradas, são entradas do pesso faz, pesos camada oculta:

        // -> Novopeso = peso(c,b) + TA * ErroC * EntradaB/SaidaB
        for (int i=0; i < totalPesos; i++) {
            pesoAtual = pesos.getPeso(i);
            posEntrada = pesos.getPosEntrada(i);
            posSaida = pesos.getPosSaida(i);
            novoPeso = pesoAtual + taxaAprendizagem * neuronios.get(posSaida).getErro() *
                    neuronios.get(posEntrada).getSaida();

            pesos.setPeso(novoPeso, i);
        }
    }
}
