package com.artificialintelligence.mlp.controller;

import com.artificialintelligence.mlp.model.CalculationParameters;
import com.artificialintelligence.mlp.model.EntradaDados;
import com.artificialintelligence.mlp.model.Neuron;
import com.artificialintelligence.mlp.model.OneHotEncoding;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.Arrays;

@RestController
public class MLPController {

    CalculationParameters calculationParameters;
    String[] classes;
    float[][] dadosOneHotEncoding;


    @PostMapping("/entrada")
    public EntradaDados entradaDados(@RequestBody EntradaDados dados) {

        // 1º - Transformar a Matriz e faz One Hot Encode
        classes = new OneHotEncoding()
                .retornarClasses(dados.getTrainingData(), dados.getCalculationParameters().getOutputLayer());
        dadosOneHotEncoding = new OneHotEncoding()
                .tratarDados(dados.getTrainingData(), dados.getCalculationParameters().getOutputLayer(), classes);
        // --> Exibição
        new OneHotEncoding().Exibicao(dadosOneHotEncoding);

        // 2 - Criar os Neuroneos
        // Os neuroneos observarão apenas as colunas de dados, que é até:
        // -> int TL_Dados = dadosOneHotEncoding.length - classes.length
        // -> Analisar até essa posição que é onde estão os dados

        return dados;
    }

    @GetMapping("/saida")
    public CalculationParameters saidaDados() {
        System.out.println(calculationParameters.getTransferFunction());
        return calculationParameters;
    }

}
