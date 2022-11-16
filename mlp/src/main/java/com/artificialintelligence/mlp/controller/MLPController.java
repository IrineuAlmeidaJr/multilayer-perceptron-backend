package com.artificialintelligence.mlp.controller;

import com.artificialintelligence.mlp.model.*;
import com.artificialintelligence.mlp.model.FuncoesTransferencia.Linear;
import com.artificialintelligence.mlp.model.FuncoesTransferencia.Logistica;
import com.artificialintelligence.mlp.model.FuncoesTransferencia.TangenteHiperbolica;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;

@RestController
public class MLPController {

    CalculationParameters calculationParameters;
    String[] classes;
    double[][] dadosOneHotEncoding;
    ArrayList<Neuronio> neuronios;
    Pesos pesos;
    ArrayList<MediaErroRede> mediaErroRedeTotal;

    @PostMapping("/entrada")
    public EntradaDados entradaDados(@RequestBody EntradaDados dados) {

        // ATENCAO FAZER DEFINIÇÃO SE é LINEAR
        int opcaoFuncaoTransferencia = dados.getCalculationParameters().getTransferFunction();



        // 1º - Transformar a Matriz e faz One Hot Encode
        classes = new OneHotEncoding()
                .retornarClasses(dados.getTrainingData(), dados.getCalculationParameters().getOutputLayer());
        dadosOneHotEncoding = new OneHotEncoding()
                .tratarDados(dados.getTrainingData(), dados.getCalculationParameters().getOutputLayer(), classes);
        // --> Exibição
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
        double taxaAprendizagem = dados.getCalculationParameters().getLearningRate();
        int totalLinhas = dadosOneHotEncoding.length;
        int totalEntradas = dadosOneHotEncoding[0].length - classes.length;
        int totalSaidas = classes.length;
        do {
            vetorErroRede  = new ArrayList<Double>();
            System.out.printf("\n\n------------- Repetição -> " + (numRepeticoes+1));
//        for (int i=0; i < totalLinhas   ; i++) { // É DESSA FORMA
            for (int i = 0; i < totalLinhas; i++) {
//                System.out.printf("\nNum Entrada -> " + (i + 1)); // Espaço TESTE
                // -> Gera neuronios de ENTRADA
                neuronios = new ArrayList<Neuronio>();
                for (int j = 0; j < totalEntradas; j++) {
                    neuronios.add(new Neuronio(0, dadosOneHotEncoding[i][j], 0));
                }

                // -> Gera CAMADA OCULTA
                for (int camadaAtual = 0; camadaAtual < numCamadaOculta; camadaAtual++) {
                    for (int numNeuronio = 0; numNeuronio < totalEntradas; numNeuronio++) {
                        neuronios.add(new Neuronio(0, 0, 0));
                    }
                }

                // -> Gera neuronios de SAIDA
                for (int saidaAtual = 0; saidaAtual < totalSaidas; saidaAtual++) {
                    neuronios.add(new Neuronio(0, 0, 0));
                }

// --------------------------------------------------------------------------------------------
                // ----------------------------------------------------------------------------
                // ----------------------------------------------------------------------------
                // ATENÇÃO  -   ATENÇÃO  -  ATENÇÃO
                // DUVIDA AQUI -> só gero os pessos uma única vez e depois disso ele vai
                // recalcular, ou toda vez que volta do 0, eu tenho fazer novos pesos
                // aleatórios....
                // -> Gera os PESOS para todas as camadas
//                if (i == 0) {
                if (sorteioPesos) { // Aqui entra só uma vez
                    pesos.inicializarPesos(totalEntradas, totalSaidas, numCamadaOculta);
                    sorteioPesos = false;
                }
                // ----------------------------------------------------------------------------
                // ----------------------------------------------------------------------------
                // ----------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------


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


                // --> OBS: O laço tem que ser dentro da Função ARRUMAR
                // -> Calcula NET e SAIDA nos Neuronios
                int totalNeuronios = neuronios.size();
                for (int posNeuronios = totalEntradas;
                     posNeuronios < totalNeuronios; posNeuronios++) {
                    if (opcaoFuncaoTransferencia == 1) {
                        calcularNetSaida(neuronios, posNeuronios, pesos, new Linear());
                    } else if (opcaoFuncaoTransferencia == 2) {
                        calcularNetSaida(neuronios, posNeuronios, pesos, new Logistica());
                    } else {
                        calcularNetSaida(neuronios, posNeuronios, pesos, new TangenteHiperbolica());
                    }
                }


                // *** Calcular ERRO das SAIDAS
                if (opcaoFuncaoTransferencia == 1) {
                    calcularErroSaida(neuronios, totalSaidas,
                            dadosOneHotEncoding[i], new Linear());
                } else if (opcaoFuncaoTransferencia == 2) {
                    calcularErroSaida(neuronios, totalSaidas,
                            dadosOneHotEncoding[i], new Logistica());
                } else {
                    calcularErroSaida(neuronios, totalSaidas,
                            dadosOneHotEncoding[i], new TangenteHiperbolica());
                }

                vetorErroRede.add(erroRede(neuronios, totalSaidas, dadosOneHotEncoding[i]));

            }

            numRepeticoes++;
            mediaErroRedeAtual = calculaMediaRedeAtual(vetorErroRede);
            mediaErroRedeTotal.add(new MediaErroRede(numRepeticoes, mediaErroRedeAtual));
            System.out.printf("\nMÉDIA ERRO DE REDE ["+numRepeticoes+"] - "+ mediaErroRedeAtual);

            // ------------------
            // --> Backpropagation
            // *** Calcular ERRO dos Neuronios INTERMEDIARIOS
            if (mediaErroRedeAtual > taxaAprendizagem) {
                if (opcaoFuncaoTransferencia == 1) {
                    calcularErroIntermediario(neuronios, pesos, new Linear(),
                            totalEntradas, totalSaidas);
                } else if (opcaoFuncaoTransferencia == 2) {
                    calcularErroIntermediario(neuronios, pesos, new Logistica(),
                            totalEntradas, totalSaidas);
                } else {
                    calcularErroIntermediario(neuronios, pesos, new TangenteHiperbolica(),
                            totalEntradas, totalSaidas);
                }
            }

            // *** PAREI AQUI ***
            // --> Fazer o calculo dos novos pesos das ARESTAS

            // Por ÚLTIMO calcular novo peso para ARESTAS se ->
            // tempMediaErroRede.get(tempMediaErroRede.size()-1) > taxaAprendizagem
            // Faz isso porque pega ultimo erro de rede calculado, se for mais, ai
            // tem refaz os calculos das arestas de pesos e atribuir novos pesos.



            // *** TIRAR DEPOIS ***
            // EXIBICAO TESTE
            int ate = neuronios.size();
            System.out.println();
            for (int l = 0; l < ate; l++) {
                System.out.printf("Net: %.2f\t| Saida: %.2f\t | Erro: %f \n",
                        neuronios.get(l).getNet(), neuronios.get(l).getSaida(),
                        neuronios.get(l).getErro());
            }

        // Coloque 1 para TESTE
        }  while (numRepeticoes < 1 &&
                mediaErroRedeAtual > taxaAprendizagem);
        // -- FORMA ABAIXO É A CERTA
//        }  while (numRepeticoes < totalRepeticoes &&
//        mediaErroRedeAtual > taxaAprendizagem);


        return dados;
    }

    public void calcularNetSaida(ArrayList<Neuronio>  neuronios, int posNeuronioAtual,
                            Pesos pesos, Linear linear) {
        Neuronio neuronioEntrada;
        double tempPeso;
        double somatorio = 0;
        int totalPesos = pesos.getAllPosSaida().size();
        int posNeuronioEntrada;

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

        neuronios.get(posNeuronioAtual).setNet(somatorio);
        // Calcula SAIDA
        double saidaNeuronio = linear.calcularFuncaoSaida(somatorio);
        neuronios.get(posNeuronioAtual).setSaida(saidaNeuronio);

    }

    public void calcularNetSaida(ArrayList<Neuronio>  neuronios, int posNeuronioAtual,
                            Pesos pesos, Logistica logistica) {
        Neuronio neuronioEntrada;
        double tempPeso;
        double somatorio = 0;
        int totalPesos = pesos.getAllPosSaida().size();
        int posNeuronioEntrada;

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

        neuronios.get(posNeuronioAtual).setNet(somatorio);
        // Calcula SAIDA
        double saidaNeuronio = logistica.calcularFuncaoSaida(somatorio);
        neuronios.get(posNeuronioAtual).setSaida(saidaNeuronio);
    }

    public void calcularNetSaida(ArrayList<Neuronio>  neuronios, int posNeuronioAtual,
                                 Pesos pesos, TangenteHiperbolica tangenteHiperbolica) {
        Neuronio neuronioEntrada;
        double tempPeso;
        double somatorio = 0;
        int totalPesos = pesos.getAllPosSaida().size();
        int posNeuronioEntrada;

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

        neuronios.get(posNeuronioAtual).setNet(somatorio);
        // Calcula SAIDA
        double saidaNeuronio = tangenteHiperbolica.calcularFuncaoSaida(somatorio);
        neuronios.get(posNeuronioAtual).setSaida(saidaNeuronio);
    }


    public void calcularErroSaida(ArrayList<Neuronio>  neuronios, int totalSaidas,
                                  double[] dados, Linear linear) {

        double objetivo, obtido, erro;
        int totalNeuronios = neuronios.size();
        System.out.println("");
        for(int atual = totalNeuronios - totalSaidas, classe = dados.length - totalSaidas;
            atual < totalNeuronios; atual++, classe++) {
            objetivo = dados[classe];
            obtido = neuronios.get(atual).getSaida();
            erro = (objetivo - obtido) * linear.derivada();
            neuronios.get(atual).setErro(erro);
        }
    }

    public void calcularErroSaida(ArrayList<Neuronio>  neuronios, int totalSaidas,
                                  double[] dados, Logistica logistica) {

        double objetivo, obtido, erro;
        int totalNeuronios = neuronios.size();
        System.out.println("");
        for(int atual = totalNeuronios - totalSaidas, classe = dados.length - totalSaidas;
            atual < totalNeuronios; atual++, classe++) {
            System.out.println("ATUAL - " + atual);
            objetivo = dados[classe];
            obtido = neuronios.get(atual).getSaida();
            erro = (objetivo - obtido) * logistica.derivada(obtido);
            neuronios.get(atual).setErro(erro);
            System.out.println("Erro - " + erro);
            System.out.println("");

        }
    }

    public void calcularErroSaida(ArrayList<Neuronio>  neuronios, int totalSaidas,
                                  double[] dados, TangenteHiperbolica tangenteHiperbolica) {

        double objetivo, obtido, erro;
        int totalNeuronios = neuronios.size();
        System.out.println("");
        for(int atual = totalNeuronios - totalSaidas, classe = dados.length - totalSaidas;
            atual < totalNeuronios; atual++, classe++) {
            objetivo = dados[classe];
            obtido = neuronios.get(atual).getSaida();
            erro = (objetivo - obtido) * tangenteHiperbolica.derivada(obtido);
            neuronios.get(atual).setErro(erro);
        }
    }

    public double erroRede(ArrayList<Neuronio>  neuronios, int totalSaidas, double[] dados) {
        int totalNeuronios = neuronios.size();
        int inicio = totalNeuronios - totalSaidas;
        double erroRede = 0;
        for (int i = inicio; i < totalNeuronios; i++) {
            erroRede += Math.pow(neuronios.get(i).getErro(), 2);
        }
        erroRede /= 2;
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

    double somatorioPesosIntermediario(ArrayList<Neuronio>  neuronios, Pesos pesos, int posInicial,
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
                somatorio += neuronioSaida.getSaida() * tempPeso;
            }
            pos++;
        }

        return somatorio;
    }

    void calcularErroIntermediario(ArrayList<Neuronio>  neuronios, Pesos pesos, Linear linear,
                                   int totalEntrada, int totalSaidas) {

        int ateNeuronio = neuronios.size() - totalSaidas;
        int totalPesos = pesos.getAllPeso().size();
        int inicioPesosIntermediario = totalEntrada * totalEntrada;
        double somatorio;
        Neuronio neuronioAtual;
        for (int posNeuronioAtual=totalEntrada;
             posNeuronioAtual < ateNeuronio; posNeuronioAtual++) {

            neuronioAtual = neuronios.get(posNeuronioAtual);

            somatorio = somatorioPesosIntermediario(neuronios, pesos, inicioPesosIntermediario,
                                        posNeuronioAtual, totalPesos);

            somatorio *= linear.derivada();
            neuronioAtual.setErro(somatorio);
        }
    }

    void calcularErroIntermediario(ArrayList<Neuronio>  neuronios, Pesos pesos, Logistica logistica,
                                   int totalEntrada, int totalSaidas) {

        int ateNeuronio = neuronios.size() - totalSaidas;
        int totalPesos = pesos.getAllPeso().size();
        int inicioPesosIntermediario = totalEntrada * totalEntrada;
        double somatorio;
        Neuronio neuronioAtual;
        for (int posNeuronioAtual=totalEntrada;
             posNeuronioAtual < ateNeuronio; posNeuronioAtual++) {

            neuronioAtual = neuronios.get(posNeuronioAtual);

            somatorio = somatorioPesosIntermediario(neuronios, pesos, inicioPesosIntermediario,
                    posNeuronioAtual, totalPesos);

            somatorio *= logistica.derivada(neuronioAtual.getSaida());
            neuronioAtual.setErro(somatorio);
        }
    }

    void calcularErroIntermediario(ArrayList<Neuronio>  neuronios, Pesos pesos,
                                   TangenteHiperbolica tangenteHiperbolica,
                                   int totalEntrada, int totalSaidas) {

        int ateNeuronio = neuronios.size() - totalSaidas;
        int totalPesos = pesos.getAllPeso().size();
        int inicioPesosIntermediario = totalEntrada * totalEntrada;
        double somatorio;
        Neuronio neuronioAtual;
        for (int posNeuronioAtual=totalEntrada;
             posNeuronioAtual < ateNeuronio; posNeuronioAtual++) {

            neuronioAtual = neuronios.get(posNeuronioAtual);

            somatorio = somatorioPesosIntermediario(neuronios, pesos, inicioPesosIntermediario,
                    posNeuronioAtual, totalPesos);

            somatorio *= tangenteHiperbolica.derivada(neuronioAtual.getSaida());
            neuronioAtual.setErro(somatorio);
        }
    }



    @GetMapping("/saida")
    public CalculationParameters saidaDados() {
        System.out.println(calculationParameters.getTransferFunction());
        return calculationParameters;
    }

}
